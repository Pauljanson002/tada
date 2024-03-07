#! /bin/bash

#CUDA_VISIBLE_DEVICES=0 python -m apps.run hydra.mode=MULTIRUN text="man" action="running" wandb_mode="disabled" model.model_change=True model.use_full_pose=False name=test training.debug=True guidance.name=zeroscope model.video=True model.num_frames=5 training.iters=10,100
CUDA_VISIBLE_DEVICES=0 python -m  apps.run --multirun text="man" action="running" wandb_mode="disabled" name=mlp_fixes training.debug=False hydra.verbose=True guidance.name=zeroscope guidance.loss_type="standard" model.video=True model.num_frames=20 training.iters=400 training.accumulate=True training.constraint_latent_weight=0.0 \
model.initialize_pose="zero" model.model_change=True data.h=256 data.w=256 model.use_cubemap=True fp16=True training.scheduler=False model.add_fake_movement=False training.set_global_time_step=True training.lr=0.05 training.guidance_scale=1000 model.vpose=True,False model.pose_mlp_args.use_tau_scale=True,False model.pose_mlp_args.use_tanh_clamp=True,False model.pose_mlp_args.tanh_scale=1.0,2.0,0.5

# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "walking" 
