#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:39:36 2022

@author: fangwei123456
@modified: acxyle

TODO:
    1. keep the writing logic of sp training script
    2. split the long text into functions, not class-based structure
"""

import argparse
import os

import torch
import yaml
from spikingjelly.activation_based import surrogate, neuron, functional

import models
import train_classify
from input_encoder_fn import (
    _population_parameters,
    _resolve_phase_period,
    _check_phase_period,
    _phase_current_parameters,
    burst_encode_image,
    latency_encode_image,
    phase_encode_image,
    poisson_encode_image,
    population_encode_image,
    quantization_encode_image,
)

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def adapt_resnet_input_layer(
    model,
    in_channels: int,
):

    """replace ResNet conv1 to accept the expanded population channels.

    keep its convolution settings, device, dtype and step mode, but initialize
    new weights; this does not transfer the old conv1 weights. must be called
    before optimizer creation so the optimizer gets the new parameters.
    """
    if not hasattr(model, 'conv1'):
        raise ValueError('Population coding currently requires a ResNet-style model.conv1.')
    old = model.conv1
    kwargs = dict(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    if hasattr(old, 'step_mode'):
        kwargs['step_mode'] = old.step_mode
    new = type(old)(**kwargs).to(device=old.weight.device, dtype=old.weight.dtype)
    new.train(old.training)
    model.conv1 = new
    return model


class SResNetTrainer(train_classify.Trainer):

    def get_args_parser(self, add_help=True):

        parser = super().get_args_parser()

        parser.add_argument('--disable-input-decay', action='store_true', help='disable weight decay')
        parser.add_argument('--neuron', type=str, default='LIF')
        parser.add_argument('--surrogate', type=str, default='ATan')
        parser.add_argument('--T', type=int, default=4)

        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")

        parser.add_argument('--data-fold-training', type=bool, default=False)
        parser.add_argument('--data-fold-number', type=int, default=5 )
        parser.add_argument('--data-fold-index', type=int, default=0)     # if None, run all experiments,

        parser.add_argument('--encoder', choices=('repeat', 'poisson', 'quantization', 'latency', 'burst', 'phase', 'population'), default='repeat')
        parser.add_argument('--population-size', type=int, default=3, help='number of population anchors, integer >= 2')
        parser.add_argument('--encoder-mean', type=float, nargs='+', default=(0.485, 0.456, 0.406))
        parser.add_argument('--encoder-std', type=float, nargs='+', default=(0.229, 0.224, 0.225))
        parser.add_argument('--latency-mode', choices=('linear', 'log'), default='linear')
        parser.add_argument('--latency-zero-silent', action='store_true')
        parser.add_argument('--burst-max-spikes', type=int, default=None, help='optional cap per pixel')
        parser.add_argument('--burst-isi-min', type=int, default=1)
        parser.add_argument('--burst-isi-max', type=int, default=8)
        parser.add_argument('--phase-period', type=int, default=None, help='oscillation period; defaults to T/4 for four complete cycles')
        parser.add_argument('--phase-tau', type=float, default=2.0, help='encoder LIF tau, independent of backbone neurons')

        parser.add_argument("--add_info", type=str, default='')
        return parser
    

    def encode_sample(self, args, x: torch.Tensor):
        """select the fn encoder; train and test use the same configuration.

        repeat keeps the original normalized input. other encoders receive
        mean/std so they can restore pixel values using the existing rule.
        """
        if args.encoder == 'repeat':
            return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)

        kwargs = dict(T=args.T, mean=args.encoder_mean, std=args.encoder_std)
        if args.encoder == 'population':
            return population_encode_image(
                x, 
                population_size=args.population_size, 
                **kwargs
                )
        if args.encoder == 'phase':
            return phase_encode_image(
                x, 
                P=args.phase_period, 
                tau=args.phase_tau, 
                **kwargs
                )
        if args.encoder == 'burst':
            return burst_encode_image(
                x, 
                max_spikes=args.burst_max_spikes,
                isi_min=args.burst_isi_min, 
                isi_max=args.burst_isi_max, 
                **kwargs,
                )
        if args.encoder == 'poisson':
            return poisson_encode_image(x, **kwargs)
        if args.encoder == 'quantization':
            return quantization_encode_image(x, **kwargs)
        if args.encoder == 'latency':
            return latency_encode_image(
                x, mode=args.latency_mode, zero_silent=args.latency_zero_silent, **kwargs,
            )
        raise ValueError(f"Unknown encoder: {args.encoder}")

    def preprocess_train_sample(self, args, x: torch.Tensor):
        return self.encode_sample(args, x)

    def preprocess_test_sample(self, args, x: torch.Tensor):
        return self.encode_sample(args, x)

    def process_model_output(self, args, y: torch.Tensor):
        return y.mean(0)  # return firing rate
        # return y

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f'_T{args.T}_{args.encoder}'

    def load_model(self, args, num_classes):
        # Check selected encoder settings before constructing the backbone.
        # YAML can bypass argparse choices, so also check the name here.
        if args.encoder not in ('repeat', 'poisson', 'quantization', 'latency', 'burst', 'phase', 'population'):
            raise ValueError(f"Unknown encoder: {args.encoder}")
        if isinstance(args.T, bool) or not isinstance(args.T, int) or args.T < 1:
            raise ValueError('T must be a positive integer.')
        if args.encoder == 'phase':
            args.phase_period = _resolve_phase_period(args.T, args.phase_period)
            _check_phase_period(args.phase_period, args.phase_tau)
            _phase_current_parameters(args.phase_period, args.phase_tau)
        if args.encoder == 'burst':
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (args.burst_isi_min, args.burst_isi_max)):
                raise ValueError('burst ISI bounds must be integers.')
            if not 1 <= args.burst_isi_min <= args.burst_isi_max:
                raise ValueError('burst requires 1 <= isi_min <= isi_max.')
            cap = args.burst_max_spikes
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
                raise ValueError('burst_max_spikes must be a positive integer or None.')
        if args.encoder == 'population':
            if getattr(args, 'population_sigma', None) is not None:
                raise ValueError('Remove population_sigma from the config; Gaussian widths are calibrated automatically.')
            _population_parameters(args.T, args.population_size)

        if args.arch in model_names:

            # --- SNN
            if any(_ in args.arch for _ in ['spik', 'sew']):

                _neuron = neuron.__dict__[f'{args.neuron}Node']
                _surrogate = surrogate.__dict__[args.surrogate]()

                if ('spiking_resnet' in args.arch) and ('td' not in args.arch):
                    model = models.__dict__[args.arch](pretrained=args.pretrained,
                                                                num_classes=num_classes,
                                                                spiking_neuron=_neuron,
                                                                surrogate_function=_surrogate,
                                                                detach_reset=True,
                                                                zero_init_residual=True)     # seems only valid for IF
                elif 'tdBN_spiking_resnet' in args.arch:
                    model = models.__dict__[args.arch](
                                                        alpha=1.,
                                                        v_threshold=1.,
                                                        num_classes=num_classes,
                                                        spiking_neuron=_neuron,
                                                        surrogate_function=_surrogate,
                                                        detach_reset=True,
                                                        zero_init_residual=False)     # seems mutually exclusive with N(0,1)
                elif 'sew_resnet' in args.arch:
                    if not args.disable_input_decay:
                        model = models.__dict__[args.arch](pretrained=args.pretrained,
                                                            num_classes=num_classes,
                                                            spiking_neuron=_neuron,
                                                            surrogate_function=_surrogate,
                                                            detach_reset=True,
                                                            cnf='ADD')
                    else:
                        model = models.__dict__[args.arch](pretrained=args.pretrained,
                                                            num_classes=num_classes,
                                                            decay_input=False,     # for neuron model with leakage
                                                            spiking_neuron=_neuron,
                                                            surrogate_function=_surrogate,
                                                            detach_reset=True,
                                                            cnf='ADD')
                elif 'spiking_vgg' in args.arch:
                    if not args.disable_input_decay:
                        model = models.__dict__[args.arch](pretrained=args.pretrained,
                                                            num_classes=num_classes,
                                                            spiking_neuron=_neuron,
                                                            surrogate_function=_surrogate,
                                                            detach_reset=True)
                    else:
                        model = models.__dict__[args.arch](pretrained=args.pretrained,
                                                            num_classes=num_classes,
                                                            decay_input=False,     # for neuron model with leakage
                                                            spiking_neuron=_neuron,
                                                            surrogate_function=_surrogate,
                                                            detach_reset=True)

                elif 'spikformer' in args.arch:
                    model = models.__dict__[args.arch](
                                                        num_classes=num_classes,
                                                        spiking_neuron=_neuron,
                                                        surrogate_function=_surrogate,
                                                        detach_reset=True)

                functional.set_step_mode(model, step_mode='m')

                if args.cupy:
                    functional.set_backend(model, 'cupy', _neuron)

            # --- ANN
            else:

                model = models.__dict__[args.arch](num_classes=num_classes)

        if args.encoder == 'population':
            if args.pretrained:
                raise ValueError('Population coding changes ResNet conv1; pretrained loading is not supported.')
            if 'resnet' not in args.arch:
                raise ValueError('Population coding in the old script currently supports ResNet architectures only.')
            adapt_resnet_input_layer(model, 3 * args.population_size)

        return model


def _parse_args(_config_parser, _parser):

    args_config, remaining = _config_parser.parse_known_args()
    _root = 'models/training_configs'
    args_config.config = os.path.join(_root, args_config.config)

    cfg = {}
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
            _parser.set_defaults(**cfg)

    # 正常解析命令行
    args = _parser.parse_args(remaining)

    # --- if yaml first
    # YAML 优先：凡是 YAML 里出现的键，覆盖回去
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    return args


if __name__ == "__main__":

    # --- usage:
    # python train_imagenet_example.py -c 'spikformer_imagenet.yml' \     # <- load general config for 224*224 img
    #                                   --arch spikformer_b_16 \
    #                                   -- dataset vgg_face

    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='spikformer_imagenet.yml', type=str, metavar='FILE')

    trainer = SResNetTrainer()
    parser = trainer.get_args_parser()
    args = _parse_args(config_parser, parser)

    args.data_path = os.path.join(args.datadir, args.dataset)     # the file name with slash can be changed after this
    args.dataset = args.dataset.split('/')[0] if '/' in args.dataset else args.dataset

    args.log_postfix = f"{args.arch}_{args.dataset}"

    if any(_ in args.arch for _ in ['spik', 'sew']):
        args.log_postfix += f"_{args.neuron}_{args.surrogate}_T{args.T}"

    if args.add_info != '':
        args.log_postfix = f'{args.add_info}_' + args.log_postfix

    args.output_dir = os.path.join(f"./logs/sp_script_{args.log_postfix}_e{args.epochs}_lr{args.lr}")

    trainer.main(args)

