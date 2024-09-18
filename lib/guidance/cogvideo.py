import cv2
import numpy as np
import torchvision
from diffusers import (AutoencoderKL, DDIMScheduler, DiffusionPipeline,CogVideoXPipeline,CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler,
                       PNDMScheduler, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer, logging

logging.set_verbosity_error()
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

logger = logging.get_logger(__name__)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class CogVideo(nn.Module):
    def __init__(
        self,
        device,
        fp16,
        vram_O,
        t_range=[0.02, 0.98],
        loss_type=None,
        weighting_strategy="sds",
    ):
        super().__init__()
        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32
        self.weighting_strategy = weighting_strategy
        self.global_time_step = None
        print(f"[INFO] loading cogvideo...")

        if "SLURM_JOB_ID" in os.environ:
            model_key = "THUDM/CogVideoX-2b"
            self.pipe = CogVideoXPipeline.from_pretrained(
                model_key, torch_dtype=self.precision_t, local_files_only=True
            ).to(self.device)
            self.scheduler = CogVideoXDPMScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="trailing"
            )
        else:
            model_key = "THUDM/CogVideoX-2b"
            self.pipe = CogVideoXPipeline.from_pretrained(
                model_key, torch_dtype=self.precision_t,
            ).to(self.device)
            self.scheduler = CogVideoXDPMScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="trailing"
            )

        self.cpu_off_load = False
        if self.cpu_off_load:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()

        self.vae = self.pipe.vae.eval()
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return text_embeddings

    def train_step(
        self,
        text_embeddings,
        pred_rgbt,
        guidance_scale=100,
        rgb_as_latents=False,
        dds_embeds=None,
        **kwargs,
    ):
        if rgb_as_latents:
            latents = pred_rgbt
        else:
            pred_rgbt = F.interpolate(
                pred_rgbt, (320, 576), mode="bilinear", align_corners=False
            )
            pred_rgbt = pred_rgbt.permute(1, 0, 2, 3)[None]
            latents = self.encode_imgs(pred_rgbt)

        if self.global_time_step is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                (latents.shape[0],),
                dtype=torch.long,
                device=self.device,
            )
        else:
            logger.debug(f"Using global time step: {self.global_time_step}")
            t = self.global_time_step

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            with torch.autocast(device_type="cuda", dtype=self.precision_t):
                noise_pred = self.unet(
                    latent_model_input, tt, encoder_hidden_states=text_embeddings
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            if dds_embeds is not None:
                with torch.autocast(device_type="cuda", dtype=self.precision_t):
                    noise_pred_dds = self.unet(
                        latent_model_input, tt, encoder_hidden_states=dds_embeds
                    ).sample
                    noise_pred_uncond_dds, noise_pred_text_dds = noise_pred_dds.chunk(2)

        if dds_embeds is not None:
            noise_pred_dds = noise_pred_uncond_dds + guidance_scale * (
                noise_pred_text_dds - noise_pred_uncond_dds
            )
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.weighting_strategy == "sds":
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

        loss = (
            0.5
            * F.mse_loss(latents, (latents - grad).detach(), reduction="sum")
            / latents.shape[0]
        )

        return loss

    def produce_latents(
        self,
        text_embeddings,
        height=320,
        width=576,
        num_inference_steps=40,
        guidance_scale=100,
        num_frames=5,
        latents=None,
    ):

        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    num_frames,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast("cuda"):
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)

                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )["sample"]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        return latents

    def encode_imgs(self, imgs, normalize=True):
        if len(imgs.shape) == 4:
            print("Image is provided instead of a video adding time = 1")
            imgs = imgs[:, :, None]

        batch_size, channels, num_frames, height, width = imgs.shape

        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
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
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )

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

        return video

    def prompt_to_video(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        num_frames=5,
        guidance_scale=7.5,
        latents=None,
    ):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        uncon_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = self.get_text_embeds(prompts)
        text_embeds = torch.cat([uncon_embeds, text_embeds])

        latents = self.produce_latents(
            text_embeds,
            height=height,
            width=width,
            latents=latents,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        imgs = self.decode_latents(latents)

        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="man")
    parser.add_argument("--action", default="bad anatomy", type=str)
    parser.add_argument("--negative", default="bad anatomy", type=str)
    parser.add_argument("-H", type=int, default=320)
    parser.add_argument("-W", type=int, default=576)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")
    zs = CogVideo(device, False, True)

    for view in ["front", "back", "side"]:
        prompt = f"a {view} view 3D rendering of {opt.subject} {opt.action}, full-body"
        vid = zs.prompt_to_video(
            prompt, height=opt.H, width=opt.W, num_inference_steps=opt.steps
        )[0]
        breakpoint()
