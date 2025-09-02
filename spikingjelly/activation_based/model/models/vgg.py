#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:10:25 2024

@author: acxyle-workstation

    rewrite from pytorch vgg, added vgg25/37/58
"""

from typing import Any, cast, Dict, List, Union

import torch
import torch.nn as nn


__all__ = [
    "VGG",
    
    "vgg5", "vgg5_bn",
    
    "vgg11", "vgg11_bn",
    "vgg13", "vgg13_bn",
    "vgg16", "vgg16_bn",
    "vgg19", "vgg19_bn",
    
    "vgg25", "vgg25_bn",
    "vgg37", "vgg37_bn",
    "vgg48", "vgg48_bn",
]


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'O': [64, 'M', 128, 128, 'M'],
    
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    
    "G": [64, 64, "M", 128, 128, "M", *[256]*6, "M", *[512]*6, "M", *[512]*6, "M"],
    "H": [64, 64, "M", 128, 128, "M", *[256]*10, "M", *[512]*10, "M", *[512]*10, "M"],
    "J": [64, 64, "M", 128, 128, "M", *[256]*15, "M", *[512]*15, "M", *[512]*11, "M"],
}


def _vgg(cfg: str, batch_norm: bool, **kwargs: Any) -> VGG:
    
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    
    return model


# -----
def vgg5(**kwargs: Any):
    return _vgg("O", False, **kwargs)

def vgg5_bn(**kwargs: Any):
    return _vgg("O", True, **kwargs)

# -----
def vgg11(**kwargs: Any) -> VGG:
    return _vgg("A", False, **kwargs)


def vgg11_bn(**kwargs: Any) -> VGG:
    return _vgg("A", True, **kwargs)


def vgg13(**kwargs: Any) -> VGG:
    return _vgg("B", False, **kwargs)


def vgg13_bn(**kwargs: Any) -> VGG:
    return _vgg("B", True, **kwargs)


def vgg16(**kwargs: Any) -> VGG:
    return _vgg("D", False, **kwargs)


def vgg16_bn(**kwargs: Any) -> VGG:
    return _vgg("D", True, **kwargs)


def vgg19(**kwargs: Any) -> VGG:
    return _vgg("E", False, **kwargs)


def vgg19_bn(**kwargs: Any) -> VGG:
    return _vgg("E", True, **kwargs)

# -----
def vgg25(**kwargs: Any) -> VGG:
    return _vgg("G", False, **kwargs)

def vgg25_bn(**kwargs: Any) -> VGG:
    return _vgg("G", True, **kwargs)

def vgg37(**kwargs: Any) -> VGG:
    return _vgg("H", False, **kwargs)

def vgg37_bn(**kwargs: Any) -> VGG:
    return _vgg("H", True, **kwargs)

def vgg48(**kwargs: Any) -> VGG:
    return _vgg("J", False, **kwargs)

def vgg48_bn(**kwargs: Any) -> VGG:
    return _vgg("J", True, **kwargs)
