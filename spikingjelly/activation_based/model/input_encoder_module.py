from torch import nn

from spikingjelly.activation_based.model import input_encoder_fn as encoder_fn


class ImageEncoder(nn.Module):
    channel_multiplier = 1
    emits_sequence = True

    def __init__(self, T=64, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.T = T
        self.mean = tuple(mean)
        self.std = tuple(std)

    def extra_repr(self):
        return f"T={self.T}, mean={self.mean}, std={self.std}"


class IdentityEncoder(nn.Module):
    channel_multiplier = 1
    emits_sequence = False

    def forward(self, x):
        return x


class RepeatEncoder(ImageEncoder):
    def forward(self, x):
        return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

    def extra_repr(self):
        return f"T={self.T}"


# Use the same functions for arithmetic, RNG draws and normalization.
# Phase and population create fresh eval-mode LIF state inside each call,
# regardless of this module's train/eval mode.
class LatencyEncoder(ImageEncoder):
    def __init__(self, T=64, mode="linear", zero_silent=False, **kwargs):
        super().__init__(T=T, **kwargs)
        self.mode = mode
        self.zero_silent = zero_silent

    def forward(self, x):
        return encoder_fn.latency_encode_image(
            x, T=self.T, mean=self.mean, std=self.std,
            mode=self.mode, zero_silent=self.zero_silent,
        )

    def extra_repr(self):
        return f"{super().extra_repr()}, mode={self.mode}, zero_silent={self.zero_silent}"


class PoissonEncoder(ImageEncoder):
    def forward(self, x):
        return encoder_fn.poisson_encode_image(x, T=self.T, mean=self.mean, std=self.std)


class QuantizationEncoder(ImageEncoder):
    def forward(self, x):
        return encoder_fn.quantization_encode_image(x, T=self.T, mean=self.mean, std=self.std)


class BurstEncoder(ImageEncoder):
    def __init__(self, T=64, max_spikes=None, isi_min=1, isi_max=8, **kwargs):
        super().__init__(T=T, **kwargs)
        self.max_spikes = max_spikes
        self.isi_min = isi_min
        self.isi_max = isi_max

    def forward(self, x):
        return encoder_fn.burst_encode_image(
            x, T=self.T, mean=self.mean, std=self.std,
            max_spikes=self.max_spikes, isi_min=self.isi_min, isi_max=self.isi_max,
        )

    def extra_repr(self):
        return (f"{super().extra_repr()}, max_spikes={self.max_spikes}, "
                f"isi_min={self.isi_min}, isi_max={self.isi_max}")


class PhaseEncoder(ImageEncoder):
    def __init__(self, T=64, P=None, tau=2.0, **kwargs):
        super().__init__(T=T, **kwargs)
        self.P = encoder_fn._resolve_phase_period(T, P)
        self.tau = tau
        encoder_fn._check_phase_period(self.P, tau)

    def forward(self, x):
        return encoder_fn.phase_encode_image(
            x, T=self.T, mean=self.mean, std=self.std, P=self.P, tau=self.tau,
        )

    def extra_repr(self):
        return f"{super().extra_repr()}, P={self.P}, tau={self.tau}"


class PopulationEncoder(ImageEncoder):
    """Gaussian currents followed by independent LIF units, as in the function."""

    def __init__(self, T=64, population_size=3, **kwargs):
        super().__init__(T=T, **kwargs)
        encoder_fn._population_parameters(T, population_size)
        self.population_size = population_size

    @property
    def channel_multiplier(self):
        return self.population_size

    def forward(self, x):
        return encoder_fn.population_encode_image(
            x, T=self.T, population_size=self.population_size,
            mean=self.mean, std=self.std,
        )

    def extra_repr(self):
        tau, widths = encoder_fn._population_parameters(self.T, self.population_size)
        return (f"{super().extra_repr()}, population_size={self.population_size}, "
                f"tau={tau}, widths={widths}")


ENCODERS = {
    "identity": IdentityEncoder,
    "repeat": RepeatEncoder,
    "latency": LatencyEncoder,
    "poisson": PoissonEncoder,
    "quantization": QuantizationEncoder,
    "burst": BurstEncoder,
    "phase": PhaseEncoder,
    "population": PopulationEncoder,
}
