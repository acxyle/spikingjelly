"""
    TODO:
        convert this code to full spikingjelly style so we can utilize the mix-pricision and many
        other techniques to reduce the GPU RAM usage and accelerate the training speed.
        Now it uses too much GPU RAM and is too slow.

    1. linear version for caifar10 passed. why not working for imagenet?
    2. conv version for imagenet working

"""


import math

import torch
import torch.nn as nn

from copy import deepcopy
from spikingjelly.activation_based import layer, surrogate, neuron, functional

from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial

__all__ = ['spikformer']

class MLP(nn.Module):
    def __init__(
        self, 
        in_features,
        hidden_features=None, 
        out_features=None, 
        drop=0.,
        spiking_neuron:callable=None, 
        **kwargs
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = layer.Conv1d(in_features, hidden_features, kernel_size=1, stride=1,bias=False)
        self.fc1_bn = layer.BatchNorm1d(hidden_features)
        self.fc1_lif = spiking_neuron(**deepcopy(kwargs))

        self.fc2_conv = layer.Conv1d(hidden_features, out_features, kernel_size=1, stride=1,bias=False)
        self.fc2_bn = layer.BatchNorm1d(out_features)
        self.fc2_lif = spiking_neuron(**deepcopy(kwargs))

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x).contiguous()
        x = self.fc2_lif(x)

        return x

class SSA(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=12, 
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0., 
        proj_drop=0., 
        sr_ratio=1,
        spiking_neuron:callable=None, 
        **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_conv1d = layer.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = layer.BatchNorm1d(dim)
        self.q_lif = spiking_neuron(**deepcopy(kwargs))

        self.k_conv1d = layer.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = layer.BatchNorm1d(dim)
        self.k_lif = spiking_neuron(**deepcopy(kwargs))

        self.v_conv1d = layer.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = layer.BatchNorm1d(dim)
        self.v_lif = spiking_neuron(**deepcopy(kwargs))
        
        self.attn_lif = spiking_neuron(v_threshold=0.5, **deepcopy(kwargs))

        self.proj_conv1d = layer.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.proj_bn = layer.BatchNorm1d(dim)
        self.proj_lif = spiking_neuron(**deepcopy(kwargs))

    def forward(self, x, res_attn):

        T, B, C, N = x.shape

        q = self.q_conv1d(x)
        q = self.q_bn(q).contiguous()
        q = self.q_lif(q)
        q = q.transpose(-1, -2).reshape(T, B, N, self.num_heads, self.dim//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k = self.k_conv1d(x)
        k = self.k_bn(k).contiguous()
        k = self.k_lif(k)
        k = k.transpose(-1, -2).reshape(T, B, N, self.num_heads, self.dim//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_conv1d(x)
        v = self.v_bn(v).contiguous()
        v = self.v_lif(v)
        v = v.transpose(-1, -2).reshape(T, B, N, self.num_heads, self.dim//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # x = k.transpose(-2,-1) @ v
        # x = (q @ x) * math.sqrt(self.dim//self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()

        x = self.attn_lif(x)
        x = self.proj_conv1d(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)

        return x, attn

class Block(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop=0., 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        sr_ratio=1,
        spiking_neuron:callable=None, 
        **kwargs
        ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = SSA(
                        dim, 
                        num_heads=num_heads, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        attn_drop=attn_drop, 
                        proj_drop=drop, 
                        sr_ratio=sr_ratio,
                        spiking_neuron=spiking_neuron, 
                        **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
                    in_features=dim, 
                    hidden_features=mlp_hidden_dim, 
                    drop=drop, 
                    spiking_neuron=spiking_neuron, 
                    **kwargs)

    def forward(self, x, res_attn):
        x_attn, _ = self.attn(x, res_attn)
        x = x + x_attn
        x = x + self.mlp(x)

        return x

class SPS(nn.Module):
    def __init__(
        self, 
        img_size_h=128, 
        img_size_w=128, 
        patch_size=4, 
        in_channels=2, 
        embed_dims=256,
        spiking_neuron:callable=None, 
        **kwargs
        ):

        super().__init__()

        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = layer.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = layer.BatchNorm2d(embed_dims//8)
        self.proj_lif = spiking_neuron(**deepcopy(kwargs))

        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = layer.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = layer.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = spiking_neuron(**deepcopy(kwargs))

        self.proj_conv2 = layer.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = layer.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = spiking_neuron(**deepcopy(kwargs))

        self.proj_conv3 = layer.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = layer.BatchNorm2d(embed_dims)
        self.proj_lif3 = spiking_neuron(**deepcopy(kwargs))

        self.rpe_conv = layer.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = layer.BatchNorm2d(embed_dims)
        self.rpe_lif = spiking_neuron(**deepcopy(kwargs))

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x)     # have some fire value
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.proj_lif1(x)
        x = self.maxpool(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.proj_lif2(x)
        x = self.maxpool(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.proj_lif3(x)
        x = self.maxpool(x)

        x_feat = x
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x)
        x = x + x_feat     # (T, B, C, H, W)
        x = x.flatten(-2)     # (T, B, C, N)
        return x

class Spikformer(nn.Module):
    def __init__(
        self,
        img_size_h=128, 
        img_size_w=128, 
        patch_size=16, 
        in_channels=2, 
        num_classes=11,
        embed_dims=[64, 128, 256], 
        num_heads=[1, 2, 4], 
        mlp_ratios=[4, 4, 4], 
        qkv_bias=False, 
        qk_scale=None,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6], 
        sr_ratios=[8, 4, 2],
        spiking_neuron:callable=None, 
        **kwargs
        ):
        
        super().__init__()
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

        block = nn.ModuleList([Block(
                                    dim=embed_dims, 
                                    num_heads=num_heads, 
                                    mlp_ratio=mlp_ratios, 
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, 
                                    drop=drop_rate, 
                                    attn_drop=attn_drop_rate, 
                                    drop_path=dpr[j],
                                    norm_layer=norm_layer, 
                                    sr_ratio=sr_ratios,
                                    spiking_neuron=spiking_neuron, 
                                    **kwargs)
                                        for j in range(depths)])
        
        self.patch_embed = patch_embed
        self.block = block

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
        x = self.patch_embed(x)
        attn = None
        for blk in self.block:
            x = blk(x, attn)
        x = x.mean(3)
        x = self.head(x)
        return x


def spikformer(pretrained=False, **kwargs):
    model = Spikformer(
        img_size_h=224, 
        img_size_w=224,
        patch_size=16, 
        embed_dims=512, 
        num_heads=8, 
        mlp_ratios=4,
        in_channels=3, 
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        depths=8, 
        sr_ratios=1,
        **kwargs
    )

    return model


if __name__ == "__main__":
    
    device = 'cpu'

    _neuron = neuron.LIFNode
    _surrogate = surrogate.ATan()
    
    model = spikformer(
                    num_classes=1000,
                    spiking_neuron=_neuron, 
                    surrogate_function=_surrogate, 
                    detach_reset=True)

    # functional.set_backend(model, 'cupy', _neuron)
    functional.set_step_mode(model, step_mode='m')

    model.to(device)
    
    x = torch.ones((4,1,3,224,224))
    x = x.to(device)
    y = model(x)

    print(y.shape)
    
    