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
import yaml
import torch
from spikingjelly.activation_based import surrogate, neuron, functional
import models, train_classify

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

class SResNetTrainer(train_classify.Trainer):

    def preprocess_train_sample(self, args, x: torch.Tensor):
        # define how to process train sample before send it to model
        # return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        return x

    def preprocess_test_sample(self, args, x: torch.Tensor):
        # define how to process test sample before send it to model
        # return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        return x
        
    def process_model_output(self, args, y: torch.Tensor):
        # return y.mean(0)  # return firing rate
        return y
        
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
        
        parser.add_argument("--add_info", type=str, default='')
        return parser

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f'_T{args.T}'

    def load_model(self, args, num_classes):
    
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
    
        return model


def _parse_args(_config_parser, _parser):

    args_config, remaining = _config_parser.parse_known_args()
    _root = 'models/training_configs'
    args_config.config = os.path.join(_root, args_config.config)
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            _parser.set_defaults(**cfg)

    args = _parser.parse_args(remaining)

    return args


if __name__ == "__main__":

    # --- usage:
    # python train_imagenet_example.py -c 'spikformer_imagenet.yml' \     # <- load general config for 224*224 img
    #                                   --arch spikformer_b_16 \
    #                                   -- dataset vgg_face

    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='resnet_imagenet.yml', type=str, metavar='FILE')
    
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

