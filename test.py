import torch

from lib.guidance.zeroscope import ZeroScope

guidance = ZeroScope("cuda", True, False, t_range=[0.02, 0.98])

import torchvision

videos_list = []

prompts = [
    "a shot of front view of a man running in the beach, full-body",
    "a shot of side view of a man running in the beach, full-body",
    "a shot of back view of a man running in the beach, full-body",
    "a shot of side view of a man running in the beach, full-body",
]

prompt_refs = [
    "a shot of front view of a man in the beach, full-body",
    "a shot of side view of a man in the beach, full-body",
    "a shot of back view of a man in the beach, full-body",
    "a shot of side view of a man in the beach, full-body",
]


text_embeds  = guidance.get_text_embeds(prompts)
empty_embeds = guidance.get_text_embeds([""])
text_embeds_dds = guidance.get_text_embeds(prompt_refs)

use_latents = False


for i in range(4):
    video_read = torchvision.io.read_video(
        f"/home/paulj/projects/TADA/4d/reference_videos/{i}.mp4"
    )[0]
    video_read = video_read.permute(0,3,1,2).float()/255
    video_read = video_read[0].repeat(video_read.shape[0],1,1,1)
    video_read = video_read.to("cuda")
    if use_latents:
        video_read = guidance.encode_imgs(video_read.permute(1, 0, 2, 3)[None])
    else:
        video_read = video_read[None]
    video_read.requires_grad = True
    videos_list.append(
        video_read
    )

optimizer = torch.optim.AdamW(videos_list,lr=25e-3)


for i in range(1000):
    if i % 10 == 0:
        print(f"Step {i}")
        for i in range(4):
            if use_latents:
                torchvision.io.write_video(
                    f"/home/paulj/projects/TADA/{i}_{use_latents}.mp4",
                    guidance.decode_latents(videos_list[i]).detach().cpu().permute(0,2,3,4,1).squeeze(0) * 255,
                    10,
                )
            else:
                torchvision.io.write_video(
                    f"/home/paulj/projects/TADA/{i}_{use_latents}.mp4",
                    videos_list[i].detach().cpu().permute(0,1,3,4,2).squeeze(0) * 255,
                    10,
                )
    for d in range(4):
        dir_text_z = [
            empty_embeds[0],
            text_embeds[d],
        ]
        dir_text_z_dds = [
            empty_embeds[0],
            text_embeds_dds[d],
        ]
        dir_text_z = torch.stack(dir_text_z)
        dir_text_z_dds = torch.stack(dir_text_z_dds)
        optimizer.zero_grad()   
        if not use_latents:
            loss = guidance.train_step(dir_text_z, videos_list[d].squeeze(0), 10, use_latents)
        else:
            loss = guidance.train_step(dir_text_z, videos_list[d], 100, use_latents)
        loss.backward()
        optimizer.step()

        # save all videos

for i in range(4):
    if use_latents:
        torchvision.io.write_video(
            f"/home/paulj/projects/TADA/{i}_{use_latents}.mp4",
            guidance.decode_latents(videos_list[i])
            .detach()
            .cpu()
            .permute(0, 2, 3, 4, 1)
            .squeeze(0)
            * 255,
            10,
        )
    else:
        torchvision.io.write_video(
            f"/home/paulj/projects/TADA/{i}_{use_latents}.mp4",
            videos_list[i].detach().cpu().permute(0, 1, 3, 4, 2).squeeze(0)
            * 255,
            10,
        )
