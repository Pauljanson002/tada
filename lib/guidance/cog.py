import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
)
from transformers import T5EncoderModel, T5Tokenizer
from torch.cuda.amp import custom_bwd, custom_fwd
from diffusers.utils import export_to_video

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        (gt_grad,) = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


class CogVideoXSDS(nn.Module):
    def __init__(
        self,
        model_id="THUDM/CogVideoX-2b",
        device="cuda",
        fp16=True,
        t_range=[0.02, 0.98],
        weighting_strategy="sds",
    ):
        super().__init__()
        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32
        self.weighting_strategy = weighting_strategy

        print(f"[INFO] Loading CogVideoX SDS...")

        # Initialize CogVideoX components
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_id, subfolder="vae", torch_dtype=self.precision_t
        ).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=self.precision_t
        ).to(device)
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=self.precision_t
        ).to(device)
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        # Freeze model parameters
        for model in [self.vae, self.text_encoder, self.transformer]:
            for param in model.parameters():
                param.requires_grad = False

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scale_factor_temporal = self.vae.config.temporal_compression_ratio

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
        guidance_scale=7.5,
        rgb_as_latents=False,
    ):
        if rgb_as_latents:
            latents = pred_rgbt
        else:
            latents = self.encode_imgs(pred_rgbt)

        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (latents.shape[0],),
            dtype=torch.long,
            device=self.device,
        )

        # Prepare rotary embeddings
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            latents.shape[-2], latents.shape[-1], latents.shape[1], self.device
        )

        # Add noise to latents
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # Predict noise
        latent_model_input = torch.cat([latents_noisy] * 2)
        timestep = t.expand(latent_model_input.shape[0])

        with torch.autocast(device_type="cuda", dtype=self.precision_t):
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=text_embeddings,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        w = (
            (1 - self.alphas[t]).view(-1, 1, 1, 1, 1)
            if self.weighting_strategy == "sds"
            else (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1, 1)
        )

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        loss = SpecifyGradient.apply(latents, grad)
        return loss

    def produce_latents(
        self,
        text_embeddings,
        height=320,
        width=576,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_frames=49,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.transformer.config.in_channels,
                    num_frames,
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                ),
                device=self.device,
                dtype=self.precision_t,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
            num_frames,
            self.device,
        )

        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = latent_model_input.permute(0, 2, 1, 3, 4)
            timestep = t.expand(latent_model_input.shape[0]).to(self.device)

            with torch.no_grad():
                noise_pred = self.transformer(
                    hidden_states=latent_model_input.to(
                        self.device, dtype=self.precision_t
                    ),
                    encoder_hidden_states=text_embeddings.to(
                        self.device, dtype=self.precision_t
                    ),
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                
            noise_pred = noise_pred.reshape(
                -1, num_frames, self.transformer.config.in_channels, latents.shape[-2], latents.shape[-1]
            ).permute(0, 2, 1, 3, 4)

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        return latents

    def encode_imgs(self, imgs):
        batch_size, channels, num_frames, height, width = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )

        with torch.cuda.amp.autocast():
            latents = self.vae.encode(imgs).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        latents = latents.reshape(
            batch_size, num_frames, -1, latents.shape[-2], latents.shape[-1]
        )
        latents = latents.permute(0, 2, 1, 3, 4)
        return latents

    @torch.cuda.amp.autocast(enabled=True)
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = latents.permute(0, 2, 1, 3, 4).reshape(-1, *latents.shape[1:])
        video = self.vae.decode(latents).sample
        video = (video / 2 + 0.5).clamp(0, 1)
        return video

    def _prepare_rotary_positional_embeddings(self, height, width, num_frames, device):
        from diffusers.models.embeddings import get_3d_rotary_pos_embed

        grid_height = height // self.transformer.config.patch_size
        grid_width = width // self.transformer.config.patch_size
        base_size_width = 720 // self.transformer.config.patch_size
        base_size_height = 480 // self.transformer.config.patch_size

        def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
            tw = tgt_width
            th = tgt_height
            h, w = src
            r = h / w
            if r > (th / tw):
                resize_height = th
                resize_width = int(round(th / h * w))
            else:
                resize_width = tw
                resize_height = int(round(tw / w * h))
            crop_top = int(round((th - resize_height) / 2.0))
            crop_left = int(round((tw - resize_width) / 2.0))
            return (crop_top, crop_left), (
                crop_top + resize_height,
                crop_left + resize_width,
            )

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            use_real=True,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin


def generate_video_from_prompt(
    prompt, output_path, num_frames=49, num_inference_steps=50, guidance_scale=7.5
):
    # Initialize the CogVideoXSDS model
    model = CogVideoXSDS(model_id="THUDM/CogVideoX-2b", device="cuda", fp16=True)

    # Generate text embeddings
    text_embeddings = model.get_text_embeds(prompt)

    # Add unconditional embeddings for classifier-free guidance
    uncond_embeddings = model.get_text_embeds("")
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Generate latents
    latents = model.produce_latents(
        text_embeddings=text_embeddings,
        height=320,
        width=576,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
    )

    # Decode latents to video frames
    video = model.decode_latents(latents)

    # Convert to uint8 and move to CPU
    video = (video * 255).round().clamp(0, 255).to(torch.uint8).cpu()

    # Rearrange dimensions to [num_frames, height, width, channels]
    video = video.permute(1, 2, 3, 0)

    # Export video
    export_to_video(video, output_path, fps=8)

    print(f"Video generated and saved to {output_path}")


if __name__ == "__main__":
    prompt = "A panda playing guitar in a bamboo forest, surrounded by other pandas clapping along"
    output_path = "output_video.mp4"

    generate_video_from_prompt(prompt, output_path)
