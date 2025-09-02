#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:12:39 2023

@author: acxyle
"""

import torch
import torchvision.transforms as transforms

import os
import numpy as np
import random
import time

import torch.utils.data
import torchvision

from torchvision.transforms.functional import InterpolationMode
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import default_collate

from spikingjelly.activation_based import  functional
    
from tv_ref_classify import presets, transforms, utils
from tv_ref_classify.sampler import RASampler

try:
    from torchvision import prototype
except ImportError:
    prototype = None


def cal_acc1_acc5(output, target):
    # define how to calculate acc1 and acc5
    acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
    return acc1, acc5


def set_deterministic(_seed_: int=6, disable_torch_deterministic=False):
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG (Random Number Generator) for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if disable_torch_deterministic:
        pass
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    
    worker_seed = torch.initial_seed() % int(np.power(2, 32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_optimizer(args, parameters):
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = None
    return optimizer


def set_lr_scheduler(args, optimizer):
    
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "step":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosa":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs)
    elif args.lr_scheduler == "exp":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        main_lr_scheduler = None
        
    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
        else:
            warmup_lr_scheduler = None
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs])
    else:
        lr_scheduler = main_lr_scheduler
        
    return lr_scheduler


def prepare_datasets_tv(args):
    
    traindir = os.path.join(args.data_path, "train")
    valdir = os.path.join(args.data_path, "val")

    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()

    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = torchvision.datasets.ImageFolder(
                                            traindir,
                                            presets.ClassificationPresetTrain(
                                                    crop_size=train_crop_size,
                                                    interpolation=interpolation,
                                                    auto_augment_policy=auto_augment_policy,
                                                    random_erase_prob=random_erase_prob,
                                             ),
                                            )
       
    print("Took", time.time() - st)

    print("Loading validation data")
    if not args.prototype:
        preprocessing = presets.ClassificationPresetEval(
                                                        crop_size=val_crop_size, 
                                                        resize_size=val_resize_size, 
                                                        interpolation=interpolation
                                                        )
    else:
        if args.weights:
            weights = prototype.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = prototype.transforms.ImageNetEval(
                                                            crop_size=val_crop_size, 
                                                            resize_size=val_resize_size, 
                                                            interpolation=interpolation
                                                            )

    dataset_val = torchvision.datasets.ImageFolder(
                                                    valdir,
                                                    preprocessing,
                                                )
    return dataset, dataset_val


def prepare_dataset_sk(args, shuffle=False, random_state=None, **kwargs):
    
    dataset = prepare_dataset_cls_base(args, )
    
    skf = StratifiedKFold(n_splits=args.kfold_number, shuffle=shuffle, random_state=random_state)
    imgs, clses = zip(*dataset.imgs)
    skfold_indices_list = [_ for _ in skf.split(imgs, clses)]

    return dataset, skfold_indices_list


def prepare_datasets_cls(args, split_ratio=0.8, command='random', **kwargs):
    """ cls, classes, assume the directory's hierarchy is like subclasses folders """
    
    dataset = prepare_dataset_cls_base(args, )

    if command == 'random':
        
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [split_ratio, 1 - split_ratio])
        
    elif command == 'constant':
        
        num_samples = len(dataset)
        train_indices = np.arange(int(num_samples*split_ratio))
        val_indices = np.arange(len(train_indices), num_samples)
        
        dataset_train = torch.utils.data.Subset(dataset, train_indices)
        dataset_val = torch.utils.data.Subset(dataset, val_indices)
        
    else:
        
        raise ValueError
        
    return dataset_train, dataset_val


def prepare_dataset_cls_base(args, ):
    dataset = torchvision.datasets.ImageFolder(root = args.data_path, 
                                               transform = presets.ClassificationPresetEval(
                                                            resize_size=args.val_resize_size, 
                                                            crop_size=args.val_crop_size, 
                                                            interpolation=InterpolationMode(args.interpolation)
                                                                                        ),)
    
    return dataset


def prepare_datasets_train(train_dir):
    """ this function only receives data_path and return dataset, no data augmention and further process, must manually change code here"""
    
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_train = torchvision.datasets.ImageFolder(
                                        root = train_dir, 
                                        transform = torchvision.transforms.Compose([
                                                        torchvision.transforms.RandomResizedCrop(224),
                                                        torchvision.transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.ToTensor(),
                                                        normalize,
                                        ]))

    return dataset_train


def prepare_datasets_val(val_dir):
    """ this function only receives data_path and return dataset, no data augmention and further process, must manually change code here"""

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_val = torchvision.datasets.ImageFolder(
                                        root = val_dir, 
                                        transform = torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize(256),
                                                    torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.ToTensor(),
                                                    normalize,
                                        ]))

    return dataset_val


def prepare_dataloader(args, train_dataset, val_dataset):
    
    train_loader, train_sampler = prepare_dataloader_train(args, train_dataset)
    val_loader, val_sampler = prepare_dataloader_val(args, val_dataset)
    
    return train_loader, val_loader, train_sampler, val_sampler


def prepare_dataloader_train(args, train_dataset):
    
    loader_g = torch.Generator()
    loader_g.manual_seed(args.seed)
    
    # ---
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(train_dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=args.seed)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=loader_g)     # --- deterministic sequence

    # ---
    collate_fn = None
    if isinstance(train_dataset, torchvision.datasets.folder.ImageFolder):
        num_classes = len(train_dataset.classes) 
    elif isinstance(train_dataset, torch.utils.data.dataset.Subset):
        num_classes = len(train_dataset.dataset.classes)
    else:
        raise RuntimeError
    
    # ---
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
    # ---
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory= not args.disable_pinmemory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )

    return train_loader, train_sampler


def prepare_dataloader_val(args, val_dataset):

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler, 
        num_workers=args.workers, 
        pin_memory= not args.disable_pinmemory,
        worker_init_fn=seed_worker
    )
    
    return val_loader, val_sampler


# --- source code from spikingjelly training script
def load_CIFAR10(args):
    # Data loading code
    print("Loading data")
    train_crop_size = args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=True,
        transform=presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        ),
    )

    print("Took", time.time() - st)

    print("Loading validation data")

    dataset_test = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=False,
        transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    )

    print("Creating data loaders")
    loader_g = torch.Generator()
    loader_g.manual_seed(args.seed)

    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
        val_sampler = torch.utils.data.SequentialSampler(dataset_test)


    return dataset, dataset_test, train_sampler, val_sampler


def load_ImageNet(args):
    # Data loading code
    traindir = os.path.join(args.data_path, "train")
    valdir = os.path.join(args.data_path, "val")
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()

    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        ),
    )
       
    print("Took", time.time() - st)

    print("Loading validation data")
   
    if not args.prototype:
        preprocessing = presets.ClassificationPresetEval(
                                                        crop_size=val_crop_size, 
                                                        resize_size=val_resize_size, 
                                                        interpolation=interpolation
                                                        )
    else:
        if args.weights:
            weights = prototype.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = prototype.transforms.ImageNetEval(
                                                            crop_size=val_crop_size, 
                                                            resize_size=val_resize_size, 
                                                            interpolation=interpolation
                                                            )

    dataset_test = torchvision.datasets.ImageFolder(
                                                    valdir,
                                                    preprocessing,
                                                )
    
    print("Creating data loaders")
    loader_g = torch.Generator()
    loader_g.manual_seed(args.seed)

    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
        val_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, val_sampler