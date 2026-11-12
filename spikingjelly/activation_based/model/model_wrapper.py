import torch
from torch import nn
from spikingjelly.activation_based import functional


class TemporalMeanDecoder(nn.Module):
    def forward(self, x):
        return x.mean(0)


class ModelWrapper(nn.Module):
    def __init__(self, input_encoder, backbone, decoder, sequence_backbone=True):
        super().__init__()
        self.input_encoder = input_encoder
        self.backbone = backbone
        self.output_decoder = decoder
        self.sequence_backbone = sequence_backbone
        self.register_load_state_dict_pre_hook(self._load_legacy_backbone)

    def forward(self, x):
        x = self.input_encoder(x)
        if self.input_encoder.emits_sequence and not self.sequence_backbone:
            x = functional.seq_to_ann_forward(x, self.backbone)
        else:
            x = self.backbone(x)
        return self.output_decoder(x)

    def _load_legacy_backbone(
        self, 
        module, 
        state_dict, 
        prefix, 
        local_metadata,
        strict, 
        missing_keys, 
        unexpected_keys, 
        error_msgs
        ):
        
        # A pre-hook also runs when this wrapper is nested inside EMA/DDP.
        if any(key.startswith(prefix + "backbone.") for key in state_dict):
            return
        backbone_keys = self.backbone.state_dict()
        if not any(prefix + key in state_dict for key in backbone_keys):
            return
        for key in list(state_dict):
            if key.startswith(prefix):
                state_dict[prefix + "backbone." + key[len(prefix):]] = state_dict.pop(key)
        # Old checkpoints have no encoder buffers. Use the explicitly selected encoder.
        for key, value in self.input_encoder.state_dict().items():
            state_dict[prefix + "input_encoder." + key] = value
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            for key in list(metadata):
                if key == prefix.rstrip(".") or key.startswith(prefix):
                    suffix = key[len(prefix):]
                    metadata[prefix + "backbone" + ("." + suffix if suffix else "")] = metadata[key]


# Reinitialize the first convolution before optimizer creation for population input.
def expand_input_channels(backbone, multiplier):
    if multiplier == 1:
        return
    for name, conv in backbone.named_modules():
        if isinstance(conv, nn.Conv2d):
            if conv.groups != 1:
                raise ValueError("Population input currently requires an ungrouped first convolution.")
            kwargs = dict(
                in_channels=conv.in_channels * multiplier,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=conv.bias is not None,
                padding_mode=conv.padding_mode,
            )
            if hasattr(conv, "step_mode"):
                kwargs["step_mode"] = conv.step_mode
            replacement = type(conv)(**kwargs).to(device=conv.weight.device, dtype=conv.weight.dtype)
            replacement.train(conv.training)
            replacement.weight.requires_grad_(conv.weight.requires_grad)
            if conv.bias is not None:
                with torch.no_grad():
                    replacement.bias.copy_(conv.bias)
                replacement.bias.requires_grad_(conv.bias.requires_grad)
            parent_name, _, child_name = name.rpartition(".")
            parent = backbone.get_submodule(parent_name) if parent_name else backbone
            setattr(parent, child_name, replacement)
            return
    raise ValueError("Cannot find the backbone input convolution.")
