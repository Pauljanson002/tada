import torch
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

self = object()
model_key = "damo-vilab/text-to-video-ms-1.7b"
self.pipe = DiffusionPipeline.from_pretrained(model_key,torch_dtype=torch.float16,variant="fp16",local_files_only=True)
self.scheduler = self.pipe.scheduler