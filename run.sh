#! /bin/bash

#CUDA_VISIBLE_DEVICES=0 python -m apps.run hydra.mode=MULTIRUN text="man" action="running" wandb_mode="disabled" model.model_change=True model.use_full_pose=False name=test training.debug=True guidance.name=zeroscope model.video=True model.num_frames=5 training.iters=10,100
CUDA_VISIBLE_DEVICES=0 python -m  apps.run --multirun text="man" action="running" wandb_mode="disabled" name=test training.debug=False hydra.verbose=True guidance.name=modelscope model.video=True model.num_frames=1 training.iters=1000 training.accumulate=True training.regularize_coeff=1.0 \
model.initialize_pose="zero" model.model_change=False data.h=256 data.w=256 model.use_cubemap=True fp16=True training.scheduler=False model.add_fake_movement=True training.set_global_time_step=True training.lr=0.01
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "walking" 
