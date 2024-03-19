#! /bin/bash

#CUDA_VISIBLE_DEVICES=0 python -m apps.run hydra.mode=MULTIRUN text="man" action="running" wandb_mode="disabled" model.model_change=True model.use_full_pose=False name=test training.debug=True guidance.name=zeroscope model.video=True model.num_frames=5 training.iters=10,100
CUDA_VISIBLE_DEVICES=0 python -m apps.run --multirun \
    text="man" \
    action="running" \
    wandb_mode="online" \
    name=get_mlp_work \
    training.debug=False \
    hydra.verbose=True \
    guidance.name=no_guidance \
    guidance.loss_type="standard" \
    model.video=True \
    model.num_frames=24 \
    training.iters=2000 \
    training.accumulate=False \
    training.constraint_latent_weight=0.0 \
    training.optim="adam" \
    model.initialize_pose="running_first_frame" \
    model.model_change=True \
    data.h=256 \
    data.w=256 \
    model.use_cubemap=False \
    fp16=True \
    training.scheduler=False \
    model.add_fake_movement=False \
    training.set_global_time_step=False \
    training.lr=0.001 \
    training.guidance_scale=100 \
    model.vpose=False \
    model.pose_mlp_args.use_tau_scale=True \
    model.pose_mlp_args.use_tanh_clamp=True \
    model.pose_mlp_args.tanh_scale=1.0 \
    training.use_ground_truth=True \
    model.pose_mlp="angle" \

# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml \
#     --text "man" \
#     --action "walking"
