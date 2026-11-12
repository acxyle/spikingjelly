import argparse
import os
from pathlib import Path

import torch
from torch import nn
import yaml

from spikingjelly.activation_based import surrogate, neuron, functional
import models
import train_classify_new
from input_encoder_module import ENCODERS
from model_wrapper import ModelWrapper, TemporalMeanDecoder, expand_input_channels


model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


class SResNetTrainer(train_classify_new.Trainer):
    def get_args_parser(self, add_help=True):

        parser = super().get_args_parser(add_help=add_help)
        parser.set_defaults(arch='spiking_resnet18')

        parser.add_argument('--disable-input-decay', action='store_true', help='disable weight decay')
        parser.add_argument('--neuron', type=str, default='LIF')
        parser.add_argument('--surrogate', type=str, default='ATan')
        parser.add_argument('--T', type=int, default=4)
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")

        parser.add_argument('--data-fold-training', type=bool, default=False)
        parser.add_argument('--data-fold-number', type=int, default=5 )
        parser.add_argument('--data-fold-index', type=int, default=0)     # if None, run all experiments,

        parser.add_argument("--add_info", type=str, default='')

        parser.add_argument("--encoder", choices=ENCODERS, default="repeat")
        parser.add_argument("--encoder-mean", type=float, nargs="+", default=(0.485, 0.456, 0.406))
        parser.add_argument("--encoder-std", type=float, nargs="+", default=(0.229, 0.224, 0.225))
        parser.add_argument("--latency-mode", choices=("linear", "log"), default="linear")
        parser.add_argument("--latency-zero-silent", action="store_true")
        
        parser.add_argument("--burst-max-spikes", type=int, default=None)
        parser.add_argument("--burst-isi-min", type=int, default=1)
        parser.add_argument("--burst-isi-max", type=int, default=8)

        parser.add_argument("--phase-period", type=int, default=None, help="oscillation period; defaults to T/4 for four complete cycles")
        parser.add_argument("--phase-tau", type=float, default=2.0)

        parser.add_argument("--population-size", type=int, default=3)

        return parser

    def preprocess_train_sample(self, args, x):
        return x

    def preprocess_test_sample(self, args, x):
        return x

    def process_model_output(self, args, y):
        return y

    def load_encoder(self, args):
        if args.encoder not in ENCODERS:
            raise ValueError(f"Unknown encoder: {args.encoder}")
        if args.encoder == "identity":
            return ENCODERS[args.encoder]()
        
        kwargs = dict(T=args.T, 
                      mean=args.encoder_mean, 
                      std=args.encoder_std
                      )
        
        if args.encoder == "latency":
            kwargs.update(mode=args.latency_mode, zero_silent=args.latency_zero_silent)
        elif args.encoder == "burst":
            kwargs.update(max_spikes=args.burst_max_spikes, isi_min=args.burst_isi_min, isi_max=args.burst_isi_max)
        elif args.encoder == "phase":
            kwargs.update(P=args.phase_period, tau=args.phase_tau)
        elif args.encoder == "population":
            kwargs.update(population_size=args.population_size)
        return ENCODERS[args.encoder](**kwargs)

    def load_model(self, args, num_classes):
        # TODO: Transformer/Spikformer integration is outside this migration.
        if "former" in args.arch or not any(name in args.arch for name in ("resnet", "resnext", "vgg")):
            raise ValueError("The new training path currently supports CNN ResNet/VGG architectures only.")
        if args.arch not in model_names:
            raise ValueError(f"Unknown architecture: {args.arch}")
        if "tdbn" in args.arch.lower():
            raise ValueError("tdBN ResNet needs a separate factory adaptation; use spiking_resnet or sew_resnet for now.")
        
        
        


        
        encoder = self.load_encoder(args)








        sequence_backbone = any(name in args.arch for name in ("spik", "sew"))
        if sequence_backbone and not encoder.emits_sequence:
            raise ValueError("Multi-step SNN backbones require a sequence encoder; use repeat instead of identity.")

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






        expand_input_channels(model, encoder.channel_multiplier)
        decoder = TemporalMeanDecoder() if encoder.emits_sequence else nn.Identity()






        
        return ModelWrapper(encoder, model, decoder, sequence_backbone=sequence_backbone)

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f"_T{args.T}_encoder_{args.encoder}"

    def main(self, args):
        # Old trainer checkpoints serialize argparse.Namespace alongside tensors.
        # Permit that type with modern torch.load's weights_only default.
        with torch.serialization.safe_globals([argparse.Namespace]):
            return super().main(args)


def parse_args(argv=None):
    trainer = SResNetTrainer()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-c", "--config", default=None)
    config_args, remaining = config_parser.parse_known_args(argv)
    parser = trainer.get_args_parser()
    cfg = {}
    if config_args.config:
        path = Path(config_args.config)
        if not path.is_file():
            path = Path(__file__).parent / "models" / "training_configs" / path
        with path.open() as stream:
            cfg = yaml.safe_load(stream) or {}
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    # Preserve the old script's YAML-over-command-line precedence.
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return trainer, args


if __name__ == "__main__":
    trainer, args = parse_args()
    args.data_path = os.path.join(args.datadir, args.dataset)
    args.dataset = args.dataset.split("/")[0]
    args.log_postfix = f"{args.arch}_{args.dataset}_{args.encoder}"
    if any(name in args.arch for name in ("spik", "sew")):
        args.log_postfix += f"_{args.neuron}_{args.surrogate}_T{args.T}"
    if args.add_info:
        args.log_postfix = f"{args.add_info}_{args.log_postfix}"
    args.output_dir = os.path.join(f"./logs/sp_script_new_{args.log_postfix}_e{args.epochs}_lr{args.lr}")
    trainer.main(args)
