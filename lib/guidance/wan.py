
import cv2
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, DiffusionPipeline
import torchvision
# suppress partial model loading warning
logging.set_verbosity_error()
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd

import sys
sys.path.append("/home/mila/p/paul.janson/scratch/workspace/Wan2.1")

from wan.modules import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
logger = logging.get_logger(__name__)


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)  # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class Wan(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98],loss_type=None,
                 weighting_strategy='sds'):
        super().__init__()
        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32
        self.weighting_strategy = weighting_strategy
        self.global_time_step = None
        print(f'[INFO] loading wan...')
        self.model = WanModel.from_pretrained("/home/mila/p/paul.janson/scratch/workspace/Wan2.1/Wan2.1-T2V-1.3B")
        self.model.eval().requires_grad_(False)
        
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path="/home/mila/p/paul.janson/scratch/workspace/Wan2.1/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path="/home/mila/p/paul.janson/scratch/workspace/Wan2.1/Wan2.1-T2V-1.3B/google/umt5-xxl",
            shard_fn=None
        )
        
        self.vae = WanVAE(
            vae_pth="/home/mila/p/paul.janson/scratch/workspace/Wan2.1/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            device=device,
        )
        self.cpu_off_load = False
        if self.cpu_off_load:
            self.pipe.enable_sequential_cpu_offload()

        for p in self.vae.model.parameters():
            p.requires_grad = False

        # Create model
        # self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        # self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )
        # TODO SJC (DDPM scheudler)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = torch.cumprod(1-self.scheduler.sigmas, dim=0)

        # self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)


    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """
        Args:
            prompt: str

        Returns:
            text_embeddings: torch.Tensor
        """
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return text_embeddings

    def train_step(self, text_embeddings, pred_rgbt, guidance_scale=100, rgb_as_latents=False,dds_embeds=None,**kwargs): # pred_rgbt: [F, 3, H, W]
        if rgb_as_latents:
            # latents = F.interpolate(pred_rgbt, (64, 64), mode='bilinear', align_corners=False)
            latents = pred_rgbt
            # latents = latents * 2 - 1
        else:
            pred_rgbt = F.interpolate(pred_rgbt, (320, 576), mode='bilinear', align_corners=False)
            pred_rgbt = pred_rgbt.permute(1, 0, 2, 3)[None]
            latents = self.encode_imgs(pred_rgbt)

        # Before : latents = torch.mean(latents, keepdim=True, dim=0) #! Todo: Why ?????

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if self.global_time_step is None:
            t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)
            #t = torch.randint(1, 100, (latents.shape[0],), dtype=torch.long, device=self.device)
        else:
            logger.debug(f"Using global time step: {self.global_time_step}")
            t = self.global_time_step

        # TODO: SJC not implemented need to check what it is and whether it helps
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            with torch.autocast(device_type="cuda",dtype=self.precision_t):
                noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            if dds_embeds is not None:
                with torch.autocast(device_type="cuda",dtype=self.precision_t):
                    noise_pred_dds = self.unet(latent_model_input, tt, encoder_hidden_states=dds_embeds).sample
                    noise_pred_uncond_dds, noise_pred_text_dds = noise_pred_dds.chunk(2)

        # perform guidance (high scale from paper!)
        if dds_embeds is not None:
            noise_pred_dds = noise_pred_uncond_dds + guidance_scale * (noise_pred_text_dds - noise_pred_uncond_dds)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        if self.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )
        if dds_embeds is not None:
            grad = w * (noise_pred - noise_pred_dds)
        else:
            grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if False:
            with torch.no_grad():
                grad_visual = self.decode_latents((latents-grad).detach())
                grad_visual = grad_visual.cpu()[0].permute(1,2,3,0)
                grad_visual = (grad_visual * 255).to(torch.uint8)
                torchvision.io.write_video("grad_visual.mp4", grad_visual, 1)

        # TODO: Do we need gradient clipping ?
        # if grad clip
        # grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        clean_sample = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, clean_sample, reduction="sum") / latents.shape[0]
        
        clean_vid = self.decode_latents(clean_sample)[0].permute(1,2,3,0).cpu().numpy() * 255
        torchvision.io.write_video("clean_vid.mp4", clean_vid, 10)
        return loss,clean_vid

    def produce_latents(self, text_embeddings, height=320, width=576, num_inference_steps=40, guidance_scale=100,num_frames=5,
                        latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, num_frames, height // 8, width // 8),
                                  device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def encode_imgs(self, imgs,normalize = True):
        # imgs: [B, 3,F, H, W]
        if len(imgs.shape) == 4:
            print("Image is provided instead of a video adding time = 1")
            imgs = imgs[:,:,None]

        batch_size,channels, num_frames,height , width = imgs.shape

        imgs = imgs.permute(0,2,1,3,4).reshape(batch_size*num_frames,channels,height,width)
        input_dtype = imgs.dtype

        if normalize:
            imgs = 2 * imgs - 1

        with torch.cuda.amp.autocast():
            posterior = self.vae.encode(imgs.to(self.precision_t)).latent_dist
            latents = posterior.sample() * self.vae.config.scaling_factor

        latents = (
            latents[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + latents.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=True)
    def decode_latents(self, latents):
        # TODO: Make decoding align with previous version
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae.decode(latents).sample
        image = (image * 2 + 0.5).clamp(0, 1)
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        return video

    def prompt_to_video(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,num_frames=5,
                    guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        uncon_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = self.get_text_embeds(prompts)  # [2, 77, 768]
        text_embeds = torch.cat([uncon_embeds, text_embeds])

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,num_frames=num_frames,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default="man")
    parser.add_argument('--action', default='bad anatomy', type=str)
    parser.add_argument('--negative', default='bad anatomy', type=str)
    # parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
    #                     help="stable diffusion version")
    # parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=320)
    parser.add_argument('-W', type=int, default=576)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=40)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')
    zs = ZeroScope(device,False,True)
    
    for view in ["front", "back", "side"]:
        prompt = f"a {view} view 3D rendering of {opt.subject} {opt.action}, full-body"
        vid = zs.prompt_to_video(prompt, height=opt.H, width=opt.W, num_inference_steps=opt.steps)[0]
        breakpoint()
    # opt.negative
    

    # visualize image
    # plt.imshow(imgs[0])
    # plt.show()
