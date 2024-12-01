import cv2
import numpy as np
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
)
import torchvision

# suppress partial model loading warning
logging.set_verbosity_error()
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from einops import rearrange
logger = logging.get_logger(__name__)


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.zeros(
            [1], device=input_tensor.device, dtype=input_tensor.dtype
        )  # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        (gt_grad,) = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
from lib.guidance.videocrafter.utils.utils import instantiate_from_config
from lib.guidance.videocrafter.scripts.evaluation.funcs import load_model_checkpoint
from lib.guidance.videocrafter.lvdm.models.samplers.ddim import DDIMSampler

class VideoCrafter(nn.Module):
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
        print(f"[INFO] loading videocrafter 2...")

        self.config = OmegaConf.load(
            "lib/guidance/videocrafter/configs/inference_t2v_512_v2.0.yaml"
        )
        self.model_config = self.config.pop("model",OmegaConf.create())
        self.model = instantiate_from_config(self.model_config)
        self.model = self.model.to(self.device)
        self.model = load_model_checkpoint(
            self.model,
            "lib/guidance/videocrafter/checkpoints/base_512_v2/model.ckpt",
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.scheduler = DDIMSampler(self.model)
        self.num_train_timesteps = self.model_config.params.timesteps
        min_step_percent = 0.02
        max_step_percent = 0.98
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.alphas = self.model.alphas_cumprod
        print("[WARNING] DDS is not implemented for Videocrafter 2")
        print("[INFO] videocrafter 2 loaded")
        self.fps = 8
        self.motion_amp_scale = 2.0

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """
        Args:
            prompt: str

        Returns:
            text_embeddings: torch.Tensor
        """
        text_embeddings = self.model.cond_stage_model.encode(prompt).to("cuda")

        return text_embeddings

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        alphas_cumprod = self.alphas
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def compute_grad_sds(
        self,
        latents,
        text_embeddings,
        t,
        guidance_scale=100,
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            cond = {
                "c_crossattn": [text_embeddings],
                "fps": torch.full((2,), self.fps, device=latents.device),
            }
            noise_pred = self.model.apply_model(
                latent_model_input,
                torch.cat([t] * 2),
                cond,
                x0=None,
                temporal_length=latents.shape[2],
            )

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.weighting_strategy == "uniform":
            w = 1
        elif self.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.weighting_strategy}"
            )

        score = noise_pred - noise
        if self.motion_amp_scale != 1.0:
            score_mean = score.mean(2, keepdim=True)
            score = score_mean + self.motion_amp_scale * (score - score_mean)
        grad = w * score
        return grad

    def train_step(
        self,
        text_embeddings,
        pred_rgbt,
        guidance_scale=100,
        rgb_as_latents=False,
        dds_embeds=None,
        **kwargs,
    ):  # pred_rgbt: [F, 3, H, W]
        if rgb_as_latents:
            # latents = F.interpolate(pred_rgbt, (64, 64), mode='bilinear', align_corners=False)
            latents = pred_rgbt
            # latents = latents * 2 - 1
        else:
            pred_rgbt = F.interpolate(
                pred_rgbt, (320, 512), mode="bilinear", align_corners=False
            )
            pred_rgbt = pred_rgbt.permute(1, 0, 2, 3)[None]
            latents = self.encode_imgs(pred_rgbt)

        # Before : latents = torch.mean(latents, keepdim=True, dim=0) #! Todo: Why ?????

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
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

        # predict the noise residual

        grad = self.compute_grad_sds(latents, text_embeddings, t, guidance_scale)

        if False:
            with torch.no_grad():
                grad_visual = self.decode_latents(grad)
                grad_visual = grad_visual.cpu()[0].permute(1, 2, 3, 0)
                grad_visual = (grad_visual * 255).to(torch.uint8)
                torchvision.io.write_video("grad_visual.mp4", grad_visual, 5)
                breakpoint()

        # TODO: Do we need gradient clipping ?
        # if grad clip
        # grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        
        clean_sample = (latents - grad).detach()
        loss = (
            0.5
            * F.mse_loss(latents, clean_sample, reduction="sum")
            / latents.shape[0]
        )
        clean_vid = self.decode_latents(clean_sample)[0].permute(1, 2, 3, 0).cpu().numpy() * 255
        return loss

    def encode_first_stage(self, x):
        if self.model.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False

        if False:#self.low_ram_vae > 0:
            vnum = self.low_ram_vae
            mask_vae = torch.randperm(x.shape[0]) < vnum
            if not mask_vae.all():
                with torch.no_grad():
                    posterior_mask = torch.cat(
                        [
                            self.model.first_stage_model.encode(
                                x[~mask_vae][i : i + 1].to(self.weights_dtype)
                            ).sample(None)
                            for i in range(x.shape[0] - vnum)
                        ],
                        dim=0,
                    )
            posterior = torch.cat(
                [
                    self.model.first_stage_model.encode(
                        x[mask_vae][i : i + 1].to(self.weights_dtype)
                    ).sample(None)
                    for i in range(vnum)
                ],
                dim=0,
            )
            encoder_posterior = torch.zeros(
                x.shape[0],
                *posterior.shape[1:],
                device=posterior.device,
                dtype=posterior.dtype,
            )
            if not mask_vae.all():
                encoder_posterior[~mask_vae] = posterior_mask
            encoder_posterior[mask_vae] = posterior
        else:
            encoder_posterior = self.model.first_stage_model.encode(
                x.to(self.precision_t)
            ).sample(None)
        results = encoder_posterior*self.model.scale_factor
        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results

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
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )["sample"]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        return latents

    # @torch.cuda.amp.autocast(enabled=True)
    # def forward_unet(
    #     self,
    #     latents,
    #     t,
    #     encoder_hidden_states,
    # ):
    #     input_dtype = latents.dtype
    #     return self.unet(
    #         latents.to(self.weights_dtype),
    #         t.to(self.weights_dtype),
    #         encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
    #     ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=True)
    def encode_imgs(self, imgs, normalize=True):
        if len(imgs.shape) == 4:
            print("Only given an image an not video")
            imgs = imgs[:, :, None]
        batch_size, channels, num_frames, height, width = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        input_dtype = imgs.dtype
        if normalize:
            imgs = imgs * 2.0 - 1.0
        latents = self.encode_first_stage(imgs)

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
        video = self.model.decode_first_stage(latents)
        video = (video * 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
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

        # Prompts -> text embeds
        uncon_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = self.get_text_embeds(prompts)  # [2, 77, 768]
        text_embeds = torch.cat([uncon_embeds, text_embeds])

        # Text embeds -> img latents
        latents = self.produce_latents(
            text_embeds,
            height=height,
            width=width,
            latents=latents,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    vc = VideoCrafter(
        device="cuda:0",
        fp16=False,
        vram_O=0,
        t_range=[0.02, 0.98],
        loss_type=None,
        weighting_strategy="sds",
    )
    imgs = torch.randn([4,3,320,512]).cuda()
    txt_embeds = vc.model.cond_stage_model.encode(["hello world",""]).to("cuda")
    vc.train_step(
        text_embeddings=txt_embeds,
        pred_rgbt=imgs,
        guidance_scale=100,
        rgb_as_latents=False,
        dds_embeds=None,
    )
