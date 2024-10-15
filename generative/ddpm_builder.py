
import logging
from typing import Dict, Tuple, Any, Optional
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import ignite.distributed as idist

from .unet import GenericUnet

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def linear_schedule(time_steps: int) -> Tuple[Tensor, Tensor, Tensor]:
    betas = torch.linspace(1e-2, 0.2, time_steps)
    alphas = 1 - betas
    cumalphas = torch.cumprod(alphas, dim=0)
    return betas, alphas, cumalphas


def cosine_schedule(time_steps: int) -> Tuple[Tensor, Tensor, Tensor]:
    t = torch.arange(0, time_steps)
    s = 0.008
    cumalphas = torch.cos(((t/time_steps + s) / (1 + s)) * (math.pi / 2)) ** 2
    def func(t): return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas_ = []
    for i in range(time_steps):
        t1 = i / time_steps
        t2 = (i + 1) / time_steps
        betas_.append(min(1 - func(t2) / func(t1), 0.999))
    betas = torch.tensor(betas_)
    alphas = 1 - betas
    return betas, alphas, cumalphas


class DiffusionModel(nn.Module):

    betas: torch.Tensor
    alphas: torch.Tensor
    cumalphas: torch.Tensor

    def __init__(self, schedule: str, time_steps: int, size_to_generate: int):
        super().__init__()

        schedule_func = {
            "linear": linear_schedule,
            "cosine": cosine_schedule
        }[schedule]

        betas, alphas, cumalphas = schedule_func(time_steps)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("cumalphas", cumalphas)

        self.size_to_generate = size_to_generate


    @property
    def time_steps(self):
        return len(self.betas)

    def q_xt_given_xtm1(self, xtm1, t: int):
        beta = self.betas[t]
        loc = torch.sqrt(1 - beta) * xtm1
        scale = torch.sqrt(beta)
        normal = torch.distributions.Normal(loc, scale)
        return normal

    def q_xt_given_x0(self, x0, t):
        cumalpha = self.cumalphas[t]
        loc = torch.sqrt(cumalpha) * x0
        scale = torch.sqrt(1 - cumalpha)
        normal = torch.distributions.Normal(loc, scale)
        return normal


class DenoisingModel(nn.Module):

    def __init__(self, diffusion: DiffusionModel, unet: GenericUnet, reduction: str):
        super().__init__()
        self.diffusion = diffusion
        self.unet = unet
        self.reduction = reduction

    @property
    def time_steps(self):
        return self.diffusion.time_steps

    def forward(self, x: torch.Tensor, image: torch.Tensor, gt: torch.Tensor = None, t: Optional[int] = None):

        if x == None:
            x = torch.randn((image.shape[0], *self.diffusion.size_to_generate)).to(idist.device())

        if gt != None:
            if x.shape[-1] == gt.shape[-1]:
                if self.reduction == 'subsample':
                    lowres_gt = gt[:, :, ::8]
            else:
                lowres_gt = gt
            
            # Linearly upsample the lowres_gt to the same size as x. Predict the full resolution gt from the lowres_gt.
            upsampled_gt = F.interpolate(lowres_gt, (x.shape[-1]), mode='linear')
        else:
            upsampled_gt = None

        if self.training:
            return self.forward_step(x, image, upsampled_gt, t)
        else:
            return self.forward_denoising(x, image, upsampled_gt, t)

    def forward_step(self, x: torch.Tensor, image: torch.Tensor, upsampled_gt, t: int):
        return self.unet(x, image, upsampled_gt, t)

    def forward_denoising(self, xt: torch.Tensor, image: torch.Tensor, upsampled_gt: Optional[torch.Tensor], init_t: Optional[int] = None):
        
        if init_t is None:
            init_t = self.time_steps - 1

        for t in range(init_t, -1, -1):
            alpha_t = self.diffusion.alphas[t]
            cumalpha_t = self.diffusion.cumalphas[t]

            t_ = torch.full(size=(xt.shape[0],), fill_value=float(t), device=xt.device)

            # Predict the noise of x_t
            pred_noise = self.unet(xt, image, upsampled_gt, t_)
            
            # Sample next x_t
            z = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
            factor = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)
            xt_mean = 1/torch.sqrt(alpha_t) * (xt - factor*pred_noise)
            xt_sigma = torch.sqrt(self.diffusion.betas[t])
            xt = xt_mean + xt_sigma*z
        
        return xt


def build_model(
        time_steps: int,
        schedule: str,
        item_shapes: Tuple[int, int, int],
        conditioning: str,
        dataset: str,
        reduction: str,
        backbone_params: Dict[str, Any]
        ) -> DenoisingModel:
    
    size_to_generate = [1, item_shapes[1][0]]
    diffusion = DiffusionModel(schedule, time_steps, size_to_generate)

    unet = GenericUnet(
        time_steps=time_steps,
        in_ch=2 if conditioning == 'concat' or 'sr' in dataset else 1,
        out_ch=1,
        condition_shape=item_shapes[0],
        conditioning=conditioning,
        **backbone_params
    )

    model = DenoisingModel(diffusion, unet, reduction)

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    num_of_parameters_condition = sum(map(torch.numel, model.unet.condition_encoder.parameters()))
    
    LOGGER.info("Trainable params: %d of which %d in condition encoder", num_of_parameters, num_of_parameters_condition)

    return model