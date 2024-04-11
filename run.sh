#! /bin/bash

#CUDA_VISIBLE_DEVICES=0 python -m apps.run hydra.mode=MULTIRUN text="man" action="running" wandb_mode="disabled" model.model_change=True model.use_full_pose=False name=test training.debug=True guidance.name=zeroscope model.video=True model.num_frames=5 training.iters=10,100
CUDA_VISIBLE_DEVICES=0 python -m apps.run --multirun \
    text="man" \
    action="running" \
    wandb_mode="disabled" \
    name=going_to_roots \
    training.debug=False \
    hydra.verbose=True \
    guidance.name=zeroscope \
    guidance.loss_type="standard" \
    model.video=True \
    model.num_frames=10 \
    model.lock_tex=True \
    model.lock_beta=True \
    model.lock_pose=False \
    training.iters=1000 \
    training.accumulate=False \
    training.constraint_latent_weight=0.0 \
    training.optim="adam" \
    training.anneal_tex_reso=False \
    training.rgb_sds=True \
    training.normal_sds=False \
    training.mean_sds=False \
    model.initialize_pose="running_first_frame" \
    model.model_change=True \
    data.h=256 \
    data.w=256 \
    model.use_cubemap=False \
    fp16=True \
    training.scheduler="lambda" \
    model.add_fake_movement=False \
    training.set_global_time_step=False \
    training.lr=5e-5 \
    training.guidance_scale=100 \
    model.vpose=False \
    model.pose_mlp_args.tau_scale=0.0 \
    model.pose_mlp_args.use_tanh_clamp=False \
    model.pose_mlp_args.tanh_scale=0.5 \
    model.pose_mlp_args.init_zero=True \
    training.use_ground_truth=False \
    training.use_landmarks=False \
    training.use_3d_landmarks=False \
    model.pose_mlp="kickstart" \


# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml \
#     --text "man" \
#     --action "walking"
