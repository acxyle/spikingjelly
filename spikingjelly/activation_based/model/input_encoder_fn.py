"""function-based image encoders and their parameter helpers.

this file only handles encoding. model input-layer changes stay in the
training script, so the encoding functions can also be used separately.
"""

import functools
import math
import warnings

import numpy as np
from scipy.optimize import linprog
import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron




def identity_encode_image(x):
    """Return the input without a temporal dimension."""
    return x


def repeat_encode_image(x, T):
    """Repeat a batch of images along a new temporal dimension."""
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)


# --- poisson encode: rate coding one
def poisson_encode_image(
    x: torch.Tensor,
    T: int,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    # x: [N, C, H, W]

    if x.min() < 0 or x.max() > 1:
        mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, x.shape[1], 1, 1)
        std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, x.shape[1], 1, 1)
        x = (x * std + mean).clamp(0, 1)

    x = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [T, N, C, H, W]

    return torch.rand_like(x).le(x).to(x)


# --- quantization encode: rate coding two
def quantization_encode_image(
    x,
    T,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
    ):

    if x.min() < 0 or x.max() > 1:
        mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, x.shape[1], 1, 1)
        std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, x.shape[1], 1, 1)
        x = (x * std + mean).clamp(0, 1)

    spike_count = torch.round(x * T)
    spike_count = spike_count.unsqueeze(0)

    t = torch.arange(T, dtype=x.dtype, device=x.device).view(T, 1, 1, 1, 1)

    encoded_image = (
        torch.floor((t + 1) * spike_count / T)
        - torch.floor(t * spike_count / T)
    )

    return encoded_image

# --- latency coding
def latency_encode_image(
    x: torch.Tensor,
    T: int,
    mode="linear",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    zero_silent=False,
    ):

    """Encode [N, C, H, W] images into [T, N, C, H, W] spikes."""
    if not isinstance(T, int) or T < 1:
        raise ValueError("T must be a positive integer")
    if mode not in ("linear", "log"):
        raise ValueError("mode must be 'linear' or 'log'")

    # Calculate spike times in float32, including under mixed precision.
    output_dtype = x.dtype
    x = x.float()

    if x.min() < 0 or x.max() > 1:
        mean = x.new_tensor(mean).view(1, -1, 1, 1)
        std = x.new_tensor(std).view(1, -1, 1, 1)
        x = x * std + mean
    x = x.clamp(0, 1)

    if mode == "linear":
        firing_time = (T - 1) * (1 - x)
    else:
        # Equivalent to:
        # (T - 1) - log1p(expm1(T - 1) * x)
        # Stable form avoids exponentiating a large positive number.
        firing_time = -torch.log(
            x + (1 - x) * math.exp(-(T - 1))
        )

    firing_time = firing_time.round().clamp(0, T - 1).long()

    spikes = F.one_hot(firing_time, num_classes=T).movedim(-1, 0)
    if zero_silent:
        spikes = spikes * (x > 0).unsqueeze(0)

    return spikes.to(dtype=output_dtype).contiguous()


# --- burst encode (intensity -> inter-spike interval)
def burst_encode_image(
    x: torch.Tensor,
    T: int = 64,
    max_spikes: int | None = None,
    isi_min: int = 1,
    isi_max: int = 8,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    if isi_max >= T:
        warnings.warn(
            "isi_max >= T: some pixels may fire only once, "
            "so their inter-spike interval cannot be represented.",
            stacklevel=2,
        )

    if x.min() < 0 or x.max() > 1:
        mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, x.shape[1], 1, 1)
        std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, x.shape[1], 1, 1)
        x = (x * std + mean).clamp(0, 1)

    pixels = x if x.dtype == torch.float64 else x.float()
    isi = (isi_min + (isi_max - isi_min) * (1 - pixels)).round().long()

    time_indices = torch.arange(T, device=x.device).view(T, 1, 1, 1, 1)
    spikes = (x > 0) & (time_indices.remainder(isi) == 0)

    if max_spikes is not None:
        spikes &= torch.div(time_indices, isi, rounding_mode="floor") < max_spikes

    return spikes.to(x)


# --- phase encode (oscillatory current -> LIF spikes)
def _check_phase_steps(T, P):
    if isinstance(T, bool) or not isinstance(T, int) or T < 1:
        raise ValueError("T must be a positive integer.")
    if isinstance(P, bool) or not isinstance(P, int) or P < 2:
        raise ValueError("P must be an integer >= 2.")
    if T % P != 0:
        raise ValueError("T must be a multiple of P to generate complete cycles.")


def _resolve_phase_period(T, P=None):
    # No explicit period: use four complete cycles.
    if P is None:
        _check_phase_steps(T, 4)
        P = T // 4
    _check_phase_steps(T, P)
    return P


def _check_phase_period(P, tau):
    if not math.isfinite(tau) or tau <= 1:
        raise ValueError("tau must be finite and > 1.")

    beta = 1 - 1 / tau
    cosine_bound = (1 - beta - beta ** 2 - beta ** 3) / (2 * beta)
    # Sufficient boundedness condition from startup firing and steady-state silence.
    if cosine_bound >= 1:
        raise ValueError(f"tau={tau} admits no P satisfying the finite-margin calibration bound.")
    if math.cos(2 * math.pi / P) <= cosine_bound:
        minimum_period = max(2, math.floor(2 * math.pi / math.acos(cosine_bound)) + 1)
        raise ValueError(f"For tau={tau}, this calibration requires P >= {minimum_period}; got P={P}.")


@functools.lru_cache(maxsize=32)
def _phase_current_parameters(P, tau):
    if P < 2 or tau <= 1:
        raise ValueError("Phase calibration requires P >= 2 and tau > 1.")

    beta = 1 - 1 / tau
    phases = np.arange(P)
    weights = beta ** (P - 1 - phases)
    rows, limits = [], []

    for x in (0.0, 1.0):
        target = (phases != 0).astype(float) * x
        currents = np.column_stack((np.full(P, x), np.ones(P), -np.cos(2 * np.pi * phases / P)))
        steady_coeff = weights @ currents / (1 - beta ** P)
        steady_offset = -(weights @ target) / (1 - beta ** P)

        # Constrain both zero-state startup and periodic steady state.
        for coeff, offset in ((np.zeros(3), 0.0), (steady_coeff, steady_offset)):
            for t in range(P):
                coeff = beta * coeff + currents[t]
                offset = beta * offset
                sign = 1 - 2 * target[t]
                rows.append(np.r_[sign * coeff, 1.0])
                limits.append(sign * (1 - offset))
                offset -= target[t]

    # Variables are a, b, A, margin; maximize the minimum threshold margin.
    result = linprog(
        [0, 0, 0, -1], A_ub=rows, b_ub=limits,
        bounds=[(0, None), (None, None), (0, None), (0, None)],
        method="highs",
    )
    if not result.success or result.x[3] <= 1e-8:
        raise ValueError(f"No finite positive-margin optimum for P={P}, tau={tau}: {result.message}")
    return tuple(result.x[:3])


def phase_encode_image(
    x: torch.Tensor,
    T: int = 64,
    P: int | None = None,
    tau: float = 2.0,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    P = _resolve_phase_period(T, P)
    _check_phase_period(P, tau)
    tau = float(tau)
    a, b, A = _phase_current_parameters(P, tau)

    if x.min() < 0 or x.max() > 1:
        mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, x.shape[1], 1, 1)
        std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, x.shape[1], 1, 1)
        x = (x * std + mean).clamp(0, 1)

    pixels = x if x.dtype == torch.float64 else x.float()
    encoder = neuron.LIFNode(
        tau=tau,
        decay_input=False,
        v_threshold=1.0,
        v_reset=None,
        step_mode="s",
        backend="torch",
    ).eval()
    baseline = a * pixels + b
    spikes = []
    for t in range(T):
        current = baseline - A * math.cos(2 * math.pi * t / P)
        spikes.append(encoder(current))

    return torch.stack(spikes).to(x)


# --- population encode
def _population_spike_count(
    current,
    T,
    tau
):
    """
    simulate one constant-current LIF, only return the spike count.

    this is the scalar version used for parameter search, not a dummy neuron.
    start from zero, threshold=1, decay_input=False, subtractive reset.
    the actual torch neuron will be checked again after the search.
    """

    beta = 1 - 1 / tau
    voltage, count = 0.0, 0
    for _ in range(T):
        charged = beta * voltage + current
        spike = charged >= 1
        voltage = charged - spike       # soft reset
        count += spike

    return count


def _population_current_interval(
    T,
    tau
):

    """
    find the current range which gives round(T/e) spikes for this tau.

    boundary(K) finds the smallest current giving at least K spikes.
    so the desired range is [boundary(K), boundary(K+1)), approximately.
    we take its middle later, not the middle of the Gaussian width range.
    """
    if isinstance(T, bool) or not isinstance(T, int) or T < 2:
        raise ValueError("Population spike coding requires an integer T >= 2.")
    if not math.isfinite(tau) or not 1 < tau <= 5:
        raise ValueError("Population tau must be finite and in (1, 5].")

    def boundary(count):
        lower, upper = 0.0, 1.0
        for _ in range(52):     # bisection precision; fewer steps may miss a narrow interval
            current = (lower + upper) / 2
            if _population_spike_count(current, T, tau) >= count:
                upper = current
            else:
                lower = current
        return upper

    target = round(T / math.e)
    return boundary(target), boundary(target + 1)


def _population_geometry(population_size):
    """
    derive anchors and checking inputs from P, for pixel range [0, 1].

    anchors are evenly placed, including both ends, so P must be >=2.
    each unit uses its own farthest distance d=max(mu, 1-mu).
    the middle target means distance d/2, not always raw pixel=0.5.
    return anchors, distances, per-unit probe inputs, and coverage inputs.
    """
    if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size < 2:
        raise ValueError("population_size must be an integer >= 2.")
    anchors = tuple(j / (population_size - 1) for j in range(population_size))
    farthest = tuple(1. if mu < .5 else 0. for mu in anchors)
    distances = tuple(abs(end - mu) for mu, end in zip(anchors, farthest))
    middle = tuple((mu + end) / 2 for mu, end in zip(anchors, farthest))

    # Between adjacent anchors, the weakest strongest response is at their
    # equal normalized distance: (x-mu_left)/d_left = (mu_right-x)/d_right.
    # With these widths this covers the valleys, not just arbitrary sample pixels.
    coverage = tuple(
        (anchors[j] * distances[j+1] + anchors[j+1] * distances[j])
        / (distances[j] + distances[j+1])
        for j in range(population_size - 1)
    )
    return anchors, distances, (anchors, middle, farthest), coverage


@functools.lru_cache(maxsize=32)
def _population_parameters(T, population_size=3):
    """
    compute tau and Gaussian widths from our spike-count conditions.

    we removed the weakest-input first-spike-exactly-at-T condition, not
    all must-fire conditions. weakest input may be silent; preferred input
    must fire every step and the whole population should not be silent.
    the earlier combined conditions needed very large tau in experiments,
    which we dont want in our setting. here we search (1, 5], nearest to 2
    first, on a 0.01 grid. this is our experiment choice, not a universal rule.

    for each unit, halfway to its farthest input should give round(T/e)
    spikes. take the feasible current midpoint and derive sigma from it.
    P=3 gives widths (s, s/2, s); other P follow the same distance rule.
    verify using actual float32 Gaussian + LIF before returning, and cache
    by (T, population_size). we dont assume suitable for all conditions.
    """
    anchor_values, distances, probes, coverage = _population_geometry(population_size)
    anchors = torch.tensor(anchor_values, dtype=torch.float32)
    # First three rows test each unit separately. Remaining rows test the
    # whole population receiving the same pixel at each coverage valley.
    pixels = torch.tensor((*probes, *((x,) * population_size for x in coverage)), dtype=torch.float32)
    candidates = sorted((i / 100 for i in range(101, 501)), key=lambda tau: (abs(tau - 2), tau))
    for tau in candidates:
        lower, upper = _population_current_interval(T, tau)
        current = (lower + upper) / 2
        if not 0 < current < 1 or upper <= lower:
            continue
        # I_mid=exp(-(d/2)^2/(2*sigma^2)); invert it for each unit.
        scale = 0.5 / math.sqrt(-2 * math.log(current))
        widths = tuple(distance * scale for distance in distances)
        sigma = torch.tensor(widths, dtype=torch.float32)
        currents = torch.exp(-(pixels - anchors).square() / (2 * sigma.square()))
        lif = neuron.LIFNode(
            tau=tau, decay_input=False, v_threshold=1., v_reset=None,
        ).eval()
        with torch.no_grad():
            counts = torch.stack([lif(currents) for _ in range(T)]).sum(0)

        # Preferred: T; middle: round(T/e); far: no more than middle.
        # Individual silence is allowed, whole-population silence is not.
        if (torch.all(counts[0] == T) and torch.all(counts[1] == round(T / math.e))
                and torch.all(counts[2] <= counts[1]) and torch.all(counts[3:].sum(1) > 0)):
            return tau, widths
    raise ValueError(f"No verified population calibration for T={T}, P={population_size} with tau in (1, 5].")


def population_encode_image(
    x: torch.Tensor,
    T: int,
    population_size: int = 3,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    ):
    """
    this implementation takes the idea from spikingjelly:encoding.PopSpikeEncoderDeterministic

    this implementation uses LIF neuron to generate spikes, while that encoder
    uses IF and learnable mean/std. ours uses fixed anchors and computed widths.

    this implementation uses pre-defined condition to compute parameter ranges.

    ---
    Encode images as binary [T, B, P*C, H, W] spikes, with fresh LIF state.

    Channels are ordered by input channel, then ascending anchor.
    P=population_size; anchors are evenly placed in [0, 1], default P=3.
    Widths are calibrated per T; peak current and threshold are both one.
    """
    anchor_values, _, _, _ = _population_geometry(population_size)
    tau, widths = _population_parameters(T, population_size)

    if x.min() < 0 or x.max() > 1:
        mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, x.shape[1], 1, 1)
        std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, x.shape[1], 1, 1)
        x = (x * std + mean).clamp(0, 1)

    pixels = x if x.dtype == torch.float64 else x.float()
    anchors = torch.tensor(anchor_values, dtype=pixels.dtype, device=x.device).view(1, 1, population_size, 1, 1)
    sigma = torch.tensor(widths, dtype=pixels.dtype, device=x.device).view(1, 1, population_size, 1, 1)
    currents = torch.exp(-(pixels.unsqueeze(2) - anchors).square() / (2 * sigma.square()))
    currents = currents.flatten(1, 2)  # [B,C,P,H,W] -> [B,C*P,H,W]

    # Fresh state per image batch, but keep membrane memory across its T steps.
    encoder = neuron.LIFNode(
        tau=tau,
        decay_input=False,
        v_threshold=1.,
        v_reset=None,
        step_mode="s", backend="torch",
    ).eval()

    return torch.stack([encoder(currents) for _ in range(T)]).to(x)
