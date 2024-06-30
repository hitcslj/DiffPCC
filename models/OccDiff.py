import torch
import torch.nn.functional as F
from tqdm import tqdm 
from einops import rearrange, repeat
from random import random
import math
from functools import partial
from torch import nn
from torch.special import expm1
from models.dit3d import DiT3D_models  
from .build import MODELS
TRUNCATED_TIME = 0.7

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def beta_linear_log_snr(t):
    return -torch.log(torch.special.expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))
@MODELS.register_module()
class OccDiff(nn.Module):
    def __init__(
            self,
            config,
            image_size: int = 72,  
            dropout: float = 0.0,  
            verbose: bool = False,   
            eps: float = 1e-6, 
            noise_schedule: str = "linear", 
    ):
        super().__init__()
        self.image_size = image_size 
        self.eps = eps
        self.verbose = verbose 
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.denoise_fn = DiT3D_models[config.model_type](pretrained=False, 
                                                    input_size=image_size, 
                                                    num_classes=55
                                                    )   

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def training_loss(self, img, coarse_voxel, *args, **kwargs):
        batch = img.shape[0]

        times = torch.zeros(
            (batch,), device=self.device).float().uniform_(0, 1)
        noise = torch.randn_like(img)
        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        # offset_noise = torch.randn(img.shape[:2], device=img.device)
        # offset_noise = rearrange(offset_noise, 'b d -> b d 1 1 1')
        # noise = noise + offset_noise * 0.05    

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise
        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.denoise_fn(noised_img, noise_level, coarse_voxel)[0].detach_()
        pred, coarse_embedder_global, coarse_embedder = self.denoise_fn(noised_img, noise_level, coarse_voxel, self_cond) 
        return F.mse_loss(pred, img), coarse_embedder_global, coarse_embedder  

    @torch.no_grad()
    def sample_with_voxel(self, coarse_voxel, steps=50, truncated_index: float = 0.0, sketch_w: float = 1.0, verbose: bool = True):
        image_size = self.image_size
        shape = (coarse_voxel.shape[0], 1, image_size, image_size, image_size) 
        batch, device = shape[0], self.device 
        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)  
        img = torch.randn(shape, device=device)
        x_start = None

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            x_zero_none = self.denoise_fn(img, noise_cond, coarse_voxel, x_start)[0]
            x_start = x_zero_none + sketch_w * (self.denoise_fn(img, noise_cond, coarse_voxel, x_start)[0] - x_zero_none)

            if time[0] < TRUNCATED_TIME: 
                x_start.sign_()

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c

            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + torch.sqrt(variance) * noise

        return img  

    