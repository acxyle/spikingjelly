#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 12:55:31 2025

@author: zk-zhou
@modified: acxyle

    refer to: https://github.com/ZK-Zhou/spikformer/tree/main
    
    The original code adopted spikingjelly ver < 0.12, but this code uses new ver
    The original code registered model for timm training script but this code does not
    This code merged standard and lite spikformer

    TODO:
       
    1. main fucntion works ✅
    2. recorver sr settings
    3. recorver learnable settings
    4. recorver pretrained weights

"""


import math

import torch
import torch.nn as nn

from copy import deepcopy
from spikingjelly.activation_based import layer, surrogate, neuron, functional
from typing import Literal

from timm.layers import to_2tuple, trunc_normal_, DropPath
import torch.nn.functional as F

# from timm.models.vision_transformer import _cfg
# from functools import partial

__all__ = [
    'Spikformer',
    'spikformer_8_512',
    'spikformer_b_16',
    'spikformer_b_32',
    'spikformer_l_16',
    'spikformer_l_32',
    'spikformer_h_14',
    'spikformer_custom',
    
    'Spikformer_lite',
    'spikformer_4_384',
    'spikformer_8_384'
    
    ]

class MLP(nn.Module):
    def __init__(
        self, 
        in_features:int,
        hidden_features:int, 
        out_features:int, 
        drop:float,
        spiking_neuron:callable=None, 
        **kwargs
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = layer.Linear(in_features, hidden_features)
        self.fc1_bn = layer.BatchNorm1d(hidden_features)
        self.fc1_lif = spiking_neuron(**deepcopy(kwargs))

        self.fc2_conv = layer.Linear(hidden_features, out_features)
        self.fc2_bn = layer.BatchNorm1d(out_features)
        self.fc2_lif = spiking_neuron(**deepcopy(kwargs))

        self.drop = layer.Dropout(drop)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        x = self.fc1_conv(x)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).contiguous()
        x = self.fc1_lif(x)
        x = self.drop(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).contiguous()
        x = self.fc2_lif(x)

        return x


def _softplus_inv(y: float, eps: float = 1e-12) -> float:
    # Softplus 的近似反函数: raw = log(exp(y) - 1)
    return math.log(max(math.exp(y) - 1.0, eps))


class SSA(nn.Module):
    
    _version = 2
    _info = "added attn_mode to control attention action"
    
    def __init__(
        self, 
        dim:int, 
        num_heads:int, 
        qkv_bias:bool, 
        qk_scale:Literal[bool, float], 
        attn_drop:float, 
        proj_drop:float, 
        # sr_ratio:int=1,     # TODO: consider to recorver sr
        attn_mode:str="qk_v",
        spiking_neuron:callable=None, 
        **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert attn_mode in ("qk_v", "q_kv")
        self.attn_mode = attn_mode
        self.dim = dim
        self.num_heads = num_heads

        # 原始初始值：保持和原代码一致
        base_scale = qk_scale or (self.dim//self.num_heads)**-0.5
        # 参数化 qk_scale: raw -> softplus(raw) >= 0
        raw_init = _softplus_inv(float(base_scale))
        self.qk_scale_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        # self.qk_scale = qk_scale or (self.dim//self.num_heads)**-0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_linear = layer.Linear(dim, dim, bias=qkv_bias)
        self.q_bn = layer.BatchNorm1d(dim)
        self.q_lif = spiking_neuron(**deepcopy(kwargs))

        self.k_linear = layer.Linear(dim, dim, bias=qkv_bias)
        self.k_bn = layer.BatchNorm1d(dim)
        self.k_lif = spiking_neuron(**deepcopy(kwargs))

        self.v_linear = layer.Linear(dim, dim, bias=qkv_bias)
        self.v_bn = layer.BatchNorm1d(dim)
        self.v_lif = spiking_neuron(**deepcopy(kwargs))
        
        self.attn_lif = spiking_neuron(v_threshold=0.5, **deepcopy(kwargs))

        self.proj_linear = layer.Linear(dim, dim, bias=False)
        self.proj_bn = layer.BatchNorm1d(dim)
        self.proj_lif = spiking_neuron(**deepcopy(kwargs))

    @property
    def qk_scale(self) -> torch.Tensor:
        # softplus 保证 qk_scale 始终为正
        return F.softplus(self.qk_scale_raw)

    def _attn_qk_v(self, q, k, v, x):
        attn = (q @ k.transpose(-1, -2)) * self.qk_scale
        attn = self.attn_drop(attn)                            
        x = attn @ v                                          
        return x, attn

    def _attn_q_kv(self, q, k, v, x):
        kv = k.transpose(-1, -2) @ v                          
        x = (q @ kv) * self.qk_scale                              
        return x, kv

    def forward(self, x, res_attn):
        T,B,N,C = x.shape

        q = self.q_linear(x)
        q = self.q_bn(q.transpose(-1, -2)).transpose(-1, -2).contiguous()
        q = self.q_lif(q)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.dim//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k = self.k_linear(x)
        k = self.k_bn(k.transpose(-1, -2)).transpose(-1, -2).contiguous()
        k = self.k_lif(k)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.dim//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_linear(x)
        v = self.v_bn(v.transpose(-1, -2)).transpose(-1, -2).contiguous()
        v = self.v_lif(v)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.dim//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        if self.attn_mode == "qk_v":
            x, attn = self._attn_qk_v(q, k, v, x)
        elif self.attn_mode == "q_kv":
            x, attn = self._attn_q_kv(q, k, v, x)

        x = x.transpose(2, 3).reshape(T,B,N,C).contiguous()

        x = self.attn_lif(x)
        x = self.proj_linear(x)
        x = self.proj_bn(x.transpose(-1, -2)).transpose(-1, -2).contiguous()
        x = self.proj_lif(x)
        x = self.proj_drop(x)

        return x, attn


class EncoderBlock(nn.Module):
    def __init__(
        self, 
        dim:int, 
        num_heads:int, 
        mlp_ratio:int, 
        qkv_bias:bool, 
        qk_scale:Literal[bool, float],
        drop:float, 
        attn_drop:float,
        drop_path:float, 
        # norm_layer=nn.LayerNorm, 
        # sr_ratio=1,
        spiking_neuron:callable=None, 
        **kwargs
        ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        # self.norm2 = norm_layer(dim)

        self.attn = SSA(
                        dim=dim, 
                        num_heads=num_heads, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        attn_drop=attn_drop, 
                        proj_drop=drop, 
                        # sr_ratio=sr_ratio,
                        spiking_neuron=spiking_neuron, 
                        **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
                    in_features=dim, 
                    hidden_features=mlp_hidden_dim, 
                    out_features=dim,
                    drop=drop, 
                    spiking_neuron=spiking_neuron, 
                    **kwargs)

    def forward(self, x, res_attn):
        x_attn, _ = self.attn(x, res_attn)
        x = x + x_attn
        x = x + self.mlp(x)

        return x


class SPS(nn.Module):
    
    _version = 2
    _info =  "full maxpool or avgp, used for normal data like ImageNet (224*224)"
    
    def __init__(
        self, 
        img_size_h:int, 
        img_size_w:int, 
        patch_size:int, 
        in_channels:int, 
        embed_dims:int,
        spiking_neuron:callable=None, 
        **kwargs
        ):
        super().__init__()
        assert patch_size in [16,32,14]

        self.image_size = [img_size_h, img_size_w]
        self.embed_dims = embed_dims
        self.patch_size = to_2tuple(patch_size)
        self.C = in_channels
        
        assert img_size_h % patch_size == 0 and img_size_w % patch_size == 0, \
            "Input {img_size_h}x{img_size_w} must be divisible by patch {patch_size}"

        self.H = img_size_h // patch_size
        self.W = img_size_w // patch_size
        self.num_patches = self.H * self.W
        
        if patch_size in [16, 32]:
            if patch_size == 16:
                chs = [in_channels, embed_dims//8, embed_dims//4, embed_dims//2, embed_dims]
            elif patch_size == 32:
                chs = [in_channels, embed_dims//16, embed_dims//8, embed_dims//4, embed_dims//2, embed_dims]

            stem_layers = []
            for i in range(int(len(chs)-1)):
                stem_layers += [
                                layer.Conv2d(chs[i], chs[i+1], kernel_size=3, stride=1, padding=1, bias=False),
                                layer.BatchNorm2d(chs[i+1]),
                                spiking_neuron(**deepcopy(kwargs)),
                                layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
                                ]
            
        if patch_size == 14:
            chs = [in_channels, embed_dims//8, embed_dims//4, embed_dims//2]
            stem_layers = []
            for i in range(int(len(chs)-1)):
                stem_layers += [
                                layer.Conv2d(chs[i], chs[i+1], kernel_size=3, stride=1, padding=1, bias=False),
                                layer.BatchNorm2d(chs[i+1]),
                                spiking_neuron(**deepcopy(kwargs)),
                                layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
                                ]
            stem_layers += [layer.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False),
                            layer.BatchNorm2d(embed_dims),
                            spiking_neuron(**deepcopy(kwargs)),
                            layer.AdaptiveAvgPool2d((16, 16))]
                
        self.conv_stem = nn.Sequential(*stem_layers)

        self.rpe = nn.Sequential(
                            layer.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False),
                            layer.BatchNorm2d(embed_dims),
                            spiking_neuron(**deepcopy(kwargs)),
                        )

    def forward(self, x):
        x = self.conv_stem(x)
        x_feat = x
        x = self.rpe(x)
        x = x + x_feat     # (T, B, C, H, W)
        x = x.flatten(-2).transpose(-1, -2)   # (T, B, H*W, C) aka (T, B, S, E)
        
        return x


class Spikformer(nn.Module):
    def __init__(
        self,
        img_size_h:int, 
        img_size_w:int, 
        patch_size:int, 
        in_channels:int, 
        num_classes:int,
        embed_dims:int, 
        num_heads:int, 
        mlp_ratios:int, 
        qkv_bias:bool, 
        depths:int, 
        qk_scale:float,
        drop_rate:float, 
        attn_drop_rate:float, 
        drop_path_rate:float, 
        # norm_layer:callable,
        # sr_ratios:int,
        spiking_neuron:callable=None, 
        **kwargs
        ):
        
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(img_size_h=img_size_h,
                            img_size_w=img_size_w,
                            patch_size=patch_size,
                            in_channels=in_channels,
                            embed_dims=embed_dims,
                            spiking_neuron=spiking_neuron, 
                            **kwargs)

        Encoder = nn.ModuleList([EncoderBlock(
                                    dim=embed_dims, 
                                    num_heads=num_heads, 
                                    mlp_ratio=mlp_ratios, 
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, 
                                    drop=drop_rate, 
                                    attn_drop=attn_drop_rate, 
                                    drop_path=dpr[j],
                                    # norm_layer=norm_layer, 
                                    # sr_ratio=sr_ratios,
                                    spiking_neuron=spiking_neuron, 
                                    **kwargs)
                                        for j in range(depths)])
        
        self.patch_embed = patch_embed
        self.Encoder = Encoder

        # classification head
        self.head = layer.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):     # --- snn type forward
        T,B,C,H,W = x.shape
        torch._assert(H == self.img_size_h, f"Wrong image height! Expected {self.img_size_h} but got {H}!")
        torch._assert(W == self.img_size_w, f"Wrong image width! Expected {self.img_size_w} but got {W}!")
        x = self.patch_embed(x)
        attn = None
        for blk in self.Encoder:
            x = blk(x, attn)
        x = x.mean(2)
        x = self.head(x)
        return x


# @register_model
def spikformer_8_512(pretrained=False, **kwargs):
    """ 
    default architecture of spikformer paper, termed as 'spikformer_8_512',
    no activate maintainance for this function, modify variables here for other architectures mentioned in the paper
    """
    model = Spikformer(
        img_size_h=224, 
        img_size_w=224,
        patch_size=16, 
        in_channels=3, 
        embed_dims=512, 
        num_heads=8, 
        mlp_ratios=4,
        qkv_bias=False,
        depths=8, 
        qk_scale=0.125,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        # sr_ratios=1,
        **kwargs
    )

    return model


# --- same as vit arch
def spikformer_b_16(pretrained=False, **kwargs):
    model = Spikformer(
        img_size_h=224, 
        img_size_w=224,
        patch_size=16, 
        in_channels=3, 
        embed_dims=768, 
        num_heads=12, 
        mlp_ratios=4,
        qkv_bias=False,
        depths=11, 
        qk_scale=None,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        # sr_ratios=1,
        **kwargs
    )

    return model


def spikformer_b_32(pretrained=False, **kwargs):
    model = Spikformer(
        img_size_h=224, 
        img_size_w=224,
        patch_size=32, 
        in_channels=3, 
        embed_dims=768, 
        num_heads=12, 
        mlp_ratios=4,
        qkv_bias=False,
        depths=11, 
        qk_scale=0.125,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        # sr_ratios=1,
        **kwargs
    )

    return model


def spikformer_l_16(pretrained=False, **kwargs):
    model = Spikformer(
        img_size_h=224, 
        img_size_w=224,
        patch_size=16, 
        in_channels=3, 
        embed_dims=1024, 
        num_heads=16, 
        mlp_ratios=4,
        qkv_bias=False,
        depths=23, 
        qk_scale=0.125,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        # sr_ratios=1,
        **kwargs
    )

    return model


def spikformer_l_32(pretrained=False, **kwargs):
    model = Spikformer(
        img_size_h=224, 
        img_size_w=224,
        patch_size=32, 
        in_channels=3, 
        embed_dims=1024, 
        num_heads=16, 
        mlp_ratios=4,
        qkv_bias=False,
        depths=23, 
        qk_scale=0.125,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        # sr_ratios=1,
        **kwargs
    )

    return model


def spikformer_h_14(pretrained=False, **kwargs):
    model = Spikformer(
        img_size_h=224, 
        img_size_w=224,
        patch_size=14, 
        in_channels=3, 
        embed_dims=1280, 
        num_heads=16, 
        mlp_ratios=4,
        qkv_bias=False,
        depths=31, 
        qk_scale=0.125,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        # sr_ratios=1,
        **kwargs
    )

    return model


def spikformer_custom(pretrained=False, **kwargs):
    return Spikformer(
                    **kwargs
                    )


class SPS_lite(nn.Module):
    
    _version = 2
    _info = "only 2 maxpool, used for small data like cifar10 (32*32)"
    
    def __init__(
        self, 
        img_size_h:int, 
        img_size_w:int, 
        patch_size:int, 
        in_channels:int, 
        embed_dims:int,
        spiking_neuron:callable=None, 
        **kwargs
        ):
        super().__init__()
        assert patch_size in [4, 8]

        self.image_size = [img_size_h, img_size_w]
        self.embed_dims = embed_dims
        self.patch_size = to_2tuple(patch_size)
        self.C = in_channels
        
        assert img_size_h % patch_size == 0 and img_size_w % patch_size == 0, \
            "Input {img_size_h}x{img_size_w} must be divisible by patch {patch_size}"

        self.H = img_size_h // patch_size
        self.W = img_size_w // patch_size
        self.num_patches = self.H * self.W
        
        if patch_size == 4:
            chs = [in_channels, embed_dims//8, embed_dims//4, embed_dims//2, embed_dims]
        elif patch_size == 8:
            chs = [in_channels, embed_dims//16, embed_dims//8, embed_dims//4, embed_dims//2, embed_dims]

        stem_layers = []
        for i in range(len(chs)-1):
            stem_layers += [
                            layer.Conv2d(chs[i], chs[i+1], kernel_size=3, stride=1, padding=1, bias=False),
                            layer.BatchNorm2d(chs[i+1]),
                            spiking_neuron(**deepcopy(kwargs)),
                            ]
            if i > 1:
                stem_layers += [layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)]
                
        self.conv_stem = nn.Sequential(*stem_layers)

        self.rpe = nn.Sequential(
                            layer.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False),
                            layer.BatchNorm2d(embed_dims),
                            spiking_neuron(**deepcopy(kwargs)),
                        )

    def forward(self, x):
        x = self.conv_stem(x)
        x_feat = x
        x = self.rpe(x)
        x = x + x_feat     # (T, B, C, H, W)
        x = x.flatten(-2).transpose(-1, -2)   # (T, B, H*W, C) aka (T, B, S, E)
        
        return x
    

class Spikformer_lite(nn.Module):
    def __init__(
        self,
        img_size_h:int, 
        img_size_w:int, 
        patch_size:int, 
        in_channels:int, 
        num_classes:int,
        embed_dims:int, 
        num_heads:int, 
        mlp_ratios:int, 
        qkv_bias:bool, 
        depths:int, 
        qk_scale:float,
        drop_rate:float, 
        attn_drop_rate:float, 
        drop_path_rate:float, 
        # norm_layer:callable,
        # sr_ratios:int,
        spiking_neuron:callable=None, 
        **kwargs
        ):
        
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS_lite(img_size_h=img_size_h,
                            img_size_w=img_size_w,
                            patch_size=patch_size,
                            in_channels=in_channels,
                            embed_dims=embed_dims,
                            spiking_neuron=spiking_neuron, 
                            **kwargs)

        Encoder = nn.ModuleList([EncoderBlock(
                                    dim=embed_dims, 
                                    num_heads=num_heads, 
                                    mlp_ratio=mlp_ratios, 
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, 
                                    drop=drop_rate, 
                                    attn_drop=attn_drop_rate, 
                                    drop_path=dpr[j],
                                    # norm_layer=norm_layer, 
                                    # sr_ratio=sr_ratios,
                                    spiking_neuron=spiking_neuron, 
                                    **kwargs)
                                        for j in range(depths)])
        
        self.patch_embed = patch_embed
        self.Encoder = Encoder

        # classification head
        self.head = layer.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):     # --- snn type forward
        T,B,C,H,W = x.shape
        torch._assert(H == self.img_size_h, f"Wrong image height! Expected {self.img_size_h} but got {H}!")
        torch._assert(W == self.img_size_w, f"Wrong image width! Expected {self.img_size_w} but got {W}!")
        x = self.patch_embed(x)
        attn = None
        for blk in self.Encoder:
            x = blk(x, attn)
        x = x.mean(2)
        x = self.head(x)
        return x


def spikformer_4_384(pretrained=False, **kwargs):
    """ 
    default architecture of spikformer paper, termed as 'spikformer_4_384',
    no activate maintainance for this function, modify variables here for other architectures mentioned in the paper
    """
    model = Spikformer_lite(
        img_size_h=32, 
        img_size_w=32,
        patch_size=4, 
        in_channels=3, 
        embed_dims=384, 
        num_heads=12, 
        mlp_ratios=4,
        qkv_bias=False,
        depths=4, 
        qk_scale=0.125,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        # sr_ratios=1,
        **kwargs
    )

    return model

def spikformer_8_384(pretrained=False, **kwargs):
    """ 
    default architecture of spikformer paper, termed as 'spikformer_4_384',
    no activate maintainance for this function, modify variables here for other architectures mentioned in the paper
    """
    model = Spikformer_lite(
        img_size_h=32, 
        img_size_w=32,
        patch_size=8, 
        in_channels=3, 
        embed_dims=384, 
        num_heads=12, 
        mlp_ratios=4,
        qkv_bias=False,
        depths=4, 
        qk_scale=0.125,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        # sr_ratios=1,
        **kwargs
    )

    return model


if __name__ == "__main__":
    
    device = 'cuda:0'

    _neuron = neuron.LIFNode
    _surrogate = surrogate.ATan()
    
    model = spikformer_b_16(
                    num_classes=1000,
                    spiking_neuron=_neuron, 
                    surrogate_function=_surrogate, 
                    detach_reset=True)

    functional.set_backend(model, 'cupy', _neuron)
    functional.set_step_mode(model, step_mode='m')

    model.to(device)
    
    x = torch.ones((4,1,3,224,224))
    x = x.to(device)
    y = model(x)

    print(y.shape)
    
    