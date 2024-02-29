import torch
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import imageio

pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, 
).to("cuda")

sd_pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, 
).to("cuda")



print("Cuda memory before running the model: ", torch.cuda.memory_allocated()/1024**3, "GB")

# freeze the model
for p in pipe.vae.parameters():
    p.requires_grad = False
for p in pipe.unet.parameters():
    p.requires_grad = False
for p in pipe.text_encoder.parameters():
    p.requires_grad = False

# freeze sd model
for p in sd_pipe.vae.parameters():
    p.requires_grad = False
for p in sd_pipe.unet.parameters():
    p.requires_grad = False 
for p in sd_pipe.text_encoder.parameters():
    p.requires_grad = False


print("Cuda memory after freezing the model: ", torch.cuda.memory_allocated()/1024**3, "GB")


x = torch.randn(16, 3, 256, 256).to("cuda")

breakpoint()