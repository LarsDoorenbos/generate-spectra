
from typing import Union
from math import pi

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SpatialTransformer, SpatialTransformerCustom


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.model.fc = nn.Identity()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.fc(x)
        
        x = rearrange(x, 'b f h w -> b (h w) f')
        return x


class SinCosEncoder(nn.Module):

    def __init__(self, out_features):
        super().__init__()
        self.num_components = out_features // 2
        self.in_features = 1
        self.out_features = self.in_features * 2 * self.num_components
        self.register_buffer("freqs", 2 ** torch.arange(self.num_components) * pi)

    def forward(self, x):
        aux = self.freqs[None, :] * x[:, None]
        sincos = torch.cat([torch.sin(aux), torch.cos(aux)], dim=-1)
        return sincos


class TimestepEmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, SpatialTransformerCustom) or isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x, emb)
        return x


class TimestepEmbedSequentialUp(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, skip_x, emb, context=None):
        for layer in self:
            if isinstance(layer, SpatialTransformerCustom) or isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x, skip_x, emb)
        return x


class Activation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.relu(x)


def get_conv_builder(dims: int, **conv_args):
    return DefaultClassBuilder(nn.Conv1d, **conv_args)


class DownBlock(nn.Module):

    def __init__(self, block_channels, time_embedding_dim, downsample = True):
        super().__init__()
        
        in_channels = block_channels[0]
        mid_channels = block_channels[1]
        out_channels = block_channels[2]

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            Activation()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            Activation()
        )
        self.time_mlp = nn.Sequential(
            Activation(),
            nn.Linear(time_embedding_dim, mid_channels)
        )

        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

        if downsample == True:
            self.down = nn.Conv1d(out_channels, out_channels, 3, 2, 0)
        else:
            self.down = nn.Identity()

    def forward(self, x, time_emb):
        aux = self.block1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None]
        time_emb = aux + time_emb

        aux = self.block2(time_emb) + self.res_conv(x)
        down = self.down(aux)

        return down, aux


class UpBlock(nn.Module):

    def __init__(self, block_channels, time_embedding_dim):
        super().__init__()
        
        in_channels = block_channels[0]
        mid_channels = block_channels[1]
        out_channels = block_channels[2]

        self.upsample = nn.ConvTranspose1d(in_channels // 2, mid_channels, 4, 2, 1)
        
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            Activation()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            Activation()
        )
        self.time_mlp = nn.Sequential(
            Activation(),
            nn.Linear(time_embedding_dim, mid_channels)
        )

        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, skip_x, time_emb):
        x = self.upsample(x)
        
        diff = skip_x.shape[-1] - x.shape[-1]
        padding = [diff // 2, diff // 2 + (diff % 2)]
        x = F.pad(x, padding)

        x = torch.cat([x, skip_x], dim=1)

        aux = self.block1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None]
        time_emb = aux + time_emb
        aux = self.block2(time_emb)

        return aux + self.res_conv(x)


class Condition:

    def __call__(self, info: dict):
        raise NotImplementedError()


class PositionalCondition(Condition):

    def __init__(self, mode='all') -> None:
        super().__init__()
        assert mode in ('all', 'none', 'first', 'last')
        self.mode = mode

    def __call__(self, info: dict):
        if self.mode == 'all':
            return True
        if self.mode == 'first' and info['rep_idx'] == 0:
            return True
        if self.mode == 'last' and info['rep_idx'] == info['repetitions'] - 1:
            return True
        else:
            return False


class DefaultClassBuilder:

    def __init__(self, cls, condition=None, **cls_args) -> None:
        super().__init__()
        self.cls = cls
        self.cls_args = cls_args
        self.condition = condition

    def __call__(self, *args, info: dict = None, **kwargs):
        # additional_params also enables overwriting exising params
        params = {**self.cls_args}
        for k, v in kwargs.items():
            params[k] = v
        if self.condition is None or self.condition(info):
            return self.cls(*args, **params)
        else:
            return None


class GenericUnet(nn.Module):

    def __init__(self, out_ch, in_ch, condition_shape, conditioning, dim, dim_mults, attention_depths, time_steps: int, repetitions=2, dims=1,
                 kernel_size: Union[int, tuple] = 3):
        super().__init__()
        time_embedding_dim = dim

        self.conditioning = conditioning
        self.time_steps = time_steps
        self.encoder = SinCosEncoder(time_embedding_dim)

        self.first_layer_t = nn.Sequential(
            nn.Linear(self.encoder.out_features, time_embedding_dim * 4),
            Activation(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )

        if self.conditioning == 'concat':
            self.condition_encoder = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
            self.condition_encoder.fc = nn.Identity()
            self.first_layer_image_enc = nn.Linear(512, 3598)
        elif self.conditioning == 'x-attention':
            self.condition_encoder = ResNetEncoder()
            self.cond_encoded_shape = self.condition_encoder(torch.rand(*condition_shape)[None]).shape

        self.dims = dims

        channels = [*map(lambda m: dim * m, dim_mults)]

        down_channels = [in_ch] + list(channels)[:-1]
        up_channels = [channels[0]] + list(channels)[:-1]
        bottom_channels = channels[-1]

        pad = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
        build_conv = get_conv_builder(dims, padding=pad, kernel_size=kernel_size)

        self.down_blocks = nn.ModuleList()
        depth = 0

        while depth < len(channels) - 1:
            block_channels = [down_channels[depth]] + repetitions*[down_channels[depth + 1]]
            down_block = [DownBlock(block_channels, time_embedding_dim)]
            
            if depth in attention_depths and self.conditioning == 'x-attention':
                num_heads = block_channels[-1] // 32
                dim_head = 32
                down_block.append(SpatialTransformerCustom(block_channels[-1], num_heads, dim_head, depth=1, context_dim=self.cond_encoded_shape[2]))
                
            self.down_blocks.append(TimestepEmbedSequential(*down_block))
            depth += 1          


        if self.conditioning == 'concat':
            block_channels = [down_channels[depth]] + (repetitions - 1)*[bottom_channels] + [up_channels[depth]]
            self.bottom_block = DownBlock(block_channels, time_embedding_dim, downsample=False)
        elif self.conditioning == 'x-attention':
            block_channels = [down_channels[depth]] + [up_channels[depth]] + [up_channels[depth]]
            self.bottom_block = TimestepEmbedSequential(DownBlock(block_channels, time_embedding_dim, downsample=False),
                                                        SpatialTransformerCustom(block_channels[-1], num_heads, dim_head, depth=1, context_dim=self.cond_encoded_shape[2], return_aux=False),
                                                        DownBlock(block_channels, time_embedding_dim, downsample=False))

        self.up_blocks = nn.ModuleList()
        while depth > 0:
            depth -= 1
            block_channels = [up_channels[depth+1] + down_channels[depth+1]] + (repetitions - 1) * [up_channels[depth + 1]] + [up_channels[depth]]
            up_block = [UpBlock(block_channels, time_embedding_dim)]

            if depth in attention_depths and self.conditioning == 'x-attention':
                num_heads = block_channels[-1] // 32
                dim_head = 32
                up_block.append(SpatialTransformer(block_channels[-1], num_heads, dim_head, depth=1, context_dim=self.cond_encoded_shape[2]))

            self.up_blocks.append(TimestepEmbedSequentialUp(*up_block))

        self.conv_cls = build_conv(in_channels=up_channels[depth], out_channels=out_ch, kernel_size=1, padding=0)

    def forward(self, spectrum, image, t):
        image_enc = self.condition_encoder(image)

        if self.conditioning == 'concat':  
            image_enc = self.first_layer_image_enc(image_enc[:, None])
            x = torch.cat([spectrum, image_enc], dim=1)
            context = None
        elif self.conditioning == 'x-attention':
            x = spectrum
            context = image_enc

        # Rescale t to the range [-1, 1]
        t = 2.0 * t / self.time_steps - 1.0
        # Positional encoder
        temb = self.encoder(t)
        temb = self.first_layer_t(temb)

        skip_xs = []
        for idx, down_block in enumerate(self.down_blocks):
            x, skip_x = down_block(x, temb, context)
            skip_xs.append(skip_x)

        x, _ = self.bottom_block(x, temb, context)

        for inv_depth, up_block in enumerate(self.up_blocks, 1):
            skip_x = skip_xs[-inv_depth]
            x = up_block(x, skip_x, temb, context)

        out_logits = self.conv_cls(x)
        return out_logits