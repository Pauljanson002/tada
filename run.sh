#! /bin/bash

#CUDA_VISIBLE_DEVICES=0 python -m apps.run hydra.mode=MULTIRUN text="man" action="running" wandb_mode="disabled" model.model_change=True model.use_full_pose=False name=test training.debug=True guidance.name=zeroscope model.video=True model.num_frames=5 training.iters=10,100
CUDA_VISIBLE_DEVICES=0 python -m  apps.run --multirun text="man" action="running" wandb_mode="online" name=vpose_prior training.debug=False guidance.name=naive model.video=True model.num_frames=5 training.iters=1000 training.accumulate=True training.regularize_coeff=0.0 \
model.initialize_pose="diving","running","running_gauss","running_first_frame" model.use_cubemap=False
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "walking" 
