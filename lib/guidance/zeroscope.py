import cv2
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, DiffusionPipeline

# suppress partial model loading warning
logging.set_verbosity_error()
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd



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


class ZeroScope(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98],
                 weighting_strategy='sds'):
        super().__init__()
        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32
        self.weighting_strategy = weighting_strategy

        print(f'[INFO] loading zeroscope...')

        # if hf_key is not None:
        #     print(f'[INFO] using hugging face custom model key: {hf_key}')
        #     model_key = hf_key
        # elif self.sd_version == '2.1':
        #     model_key = "stabilityai/stable-diffusion-2-1-base"
        # elif self.sd_version == '2.0':
        #     model_key = "stabilityai/stable-diffusion-2-base"
        # elif self.sd_version == '1.5':
        #     model_key = "runwayml/stable-diffusion-v1-5"
        # else:
        #     raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        if "SLURM_JOB_ID" in os.environ:
            model_key = "cerspense/zeroscope_v2_576w"
            self.pipe = DiffusionPipeline.from_pretrained(model_key,torch_dtype=self.precision_t,local_files_only=True).to(self.device)
            self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler",torch_dtype=torch.float16,local_files_only=True)
        else:
            model_key = "cerspense/zeroscope_v2_576w"
            self.pipe = DiffusionPipeline.from_pretrained(model_key,torch_dtype=self.precision_t).to(self.device)
            self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler",torch_dtype=torch.float16)
        
        self.cpu_off_load = False
        if self.cpu_off_load:
            self.pipe.enable_sequential_cpu_offload()
        
        self.vae = self.pipe.vae.eval()
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet.eval()
        
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.unet.parameters():
            p.requires_grad = False

        # Create model
        # self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        # self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        
        #TODO SJC (DDPM scheudler)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

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

    def train_step(self, text_embeddings, pred_rgbt, guidance_scale=100, rgb_as_latents=False,**kwargs): # pred_rgbt: [F, 3, H, W]
        if rgb_as_latents:
            latents = F.interpolate(pred_rgbt, (64, 64), mode='bilinear', align_corners=False)
            latents = latents * 2 - 1
        else:
            pred_rgbt = F.interpolate(pred_rgbt, (256, 256), mode='bilinear', align_corners=False)
            pred_rgbt = pred_rgbt.permute(1, 0, 2, 3)[None]
            latents = self.encode_imgs(pred_rgbt)
        
        # Before : latents = torch.mean(latents, keepdim=True, dim=0)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

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
        # perform guidance (high scale from paper!)
        
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

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        # TODO: Do we need gradient clipping ? 
        # if grad clip
        # grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * F.mse_loss(latents, (latents - grad).detach(), reduction="sum") / latents.shape[0]

        return loss

    # def train_step_perpneg(self, text_embeddings, weights, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
    #                        save_guidance_path=None):
    #     B = pred_rgb.shape[0]
    #     K = (text_embeddings.shape[0] // B) - 1  # maximum number of prompts

    #     if as_latent:
    #         latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
    #     else:
    #         # interp to 512x512 to be fed into vae.
    #         pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
    #         # encode image into latents with vae, requires grad!
    #         latents = self.encode_imgs(pred_rgb_512)

    #     # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
    #     t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

    #     # predict the noise residual with unet, NO grad!
    #     with torch.no_grad():
    #         # add noise
    #         noise = torch.randn_like(latents)
    #         latents_noisy = self.scheduler.add_noise(latents, noise, t)
    #         # pred noise
    #         latent_model_input = torch.cat([latents_noisy] * (1 + K))
    #         tt = torch.cat([t] * (1 + K))
    #         unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

    #         # perform guidance (high scale from paper!)
    #         noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
    #         delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
    #         noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds,
    #                                                                                             weights, B)

    #     # w(t), sigma_t^2
    #     w = (1 - self.alphas[t])
    #     grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
    #     grad = torch.nan_to_num(grad)

    #     if save_guidance_path:
    #         with torch.no_grad():
    #             if as_latent:
    #                 pred_rgb_512 = self.decode_latents(latents)

    #             # visualize predicted denoised image
    #             # The following block of code is equivalent to `predict_start_from_noise`...
    #             # see zero123_utils.py's version for a simpler implementation.
    #             alphas = self.scheduler.alphas.to(latents)
    #             total_timesteps = self.max_step - self.min_step + 1
    #             index = total_timesteps - t.to(latents.device) - 1
    #             b = len(noise_pred)
    #             a_t = alphas[index].reshape(b, 1, 1, 1).to(self.device)
    #             sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
    #             sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b, 1, 1, 1)).to(self.device)
    #             pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()  # current prediction for x_0
    #             result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

    #             # visualize noisier image
    #             result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

    #             # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
    #             viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image], dim=0)
    #             save_image(viz_images, save_guidance_path)

    #     # loss = SpecifyGradient.apply(latents, grad)
    #     loss = 0.5 * F.mse_loss(latents.float(), (latents - grad).detach(), reduction='sum') / latents.shape[0]

    #     return loss

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

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

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
    
    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(self, latents):
        # TODO: Make decoding align with previous version
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae.decode(latents).sample
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
        video = video.float()
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
