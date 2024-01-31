import torch
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload(gpu_id=0)

negative_prompt = "unrealistic,ugly,bad anatomy,not real,not realistic,not human,not a person,not a human"
directions = ["front","back","side"]
for i, dirn in enumerate(directions):
    print(f'Working on file {dirn}')
    action = "running"
    subject = "man"
    prompt = f"a shot of {dirn} view of a {subject} {action} in the beach"
    video_frames = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
    video_path = export_to_video(video_frames, output_video_path=f'{dirn}.mp4')