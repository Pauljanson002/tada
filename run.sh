#! /bin/bash

#CUDA_VISIBLE_DEVICES=0 python -m apps.run hydra.mode=MULTIRUN text="man" action="running" wandb_mode="disabled" model.model_change=True model.use_full_pose=False name=test training.debug=True guidance.name=zeroscope model.video=True model.num_frames=5 training.iters=10,100
CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="disabled" name=test training.debug=False guidance.name=naive model.video=True model.num_frames=5 training.iters=400 training.accumulate=True training.regularize_coeff=0.0 \
model.initialize_pose="diving" model.use_cubemap=False
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "walking" 
