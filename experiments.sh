#! /bin/bash
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --multirun text="man" action="running" wandb_mode="online" model.model_change=True model.use_full_pose=False name=rgb_normal_cube training.debug=False guidance.name=zeroscope model.video=True model.num_frames=10 training.iters=1000 training.accumulate=True training.regularize_coeff=1e3 training.rgb_sds=True training.normal_sds=False model.use_cubemap=True
# CUDA_VISIBLE_DEVICES=1 python -m apps.run --multirun text="man" action="running" wandb_mode="online" model.model_change=True model.use_full_pose=False name=rgb_normal_cube training.debug=False guidance.name=zeroscope model.video=True model.num_frames=10 training.iters=1000 training.accumulate=True training.regularize_coeff=1e3 training.rgb_sds=False training.normal_sds=True model.use_cubemap=True
# CUDA_VISIBLE_DEVICES=2 python -m apps.run --multirun text="man" action="running" wandb_mode="online" model.model_change=True model.use_full_pose=False name=rgb_normal_cube training.debug=False guidance.name=zeroscope model.video=True model.num_frames=10 training.iters=1000 training.accumulate=True training.regularize_coeff=1e3 training.rgb_sds=False training.normal_sds=True model.use_cubemap=False

# CUDA_VISIBLE_DEVICES=0 python -m  apps.run --multirun text="man" action="running with rythemic leg motion","lifting arm to the head" wandb_mode="disabled" name=global_timestep_fake_motion training.debug=True guidance.name=zeroscope,modelscope model.video=True model.num_frames=20 training.iters=1000 training.accumulate=True \
# model.model_change=False,True model.add_fake_movement=True,False training.set_global_time_step=True,False

# CUDA_VISIBLE_DEVICES=0 python -m  apps.run --multirun text="man" action="running","lifting his arm up to the head","bending and touching his toes","punching" wandb_mode="online" name=sd_std_sds training.debug=False guidance.name=sd guidance.loss_type="standard" model.video=False model.num_frames=1 training.iters=1000 training.accumulate=True training.constraint_latent_weight=0.1,1.0,2.0 model.initialize_pose="zero" model.model_change=False data.h=256 data.w=256 model.use_cubemap=True fp16=True training.scheduler=False model.add_fake_movement=True training.set_global_time_step=True training.lr=0.01,0.05,0.03,0.1 training.guidance_scale=100,60,40,7.5
# CUDA_VISIBLE_DEVICES=1 python -m  apps.run --multirun text="man" action="running","lifting his arm up to the head","bending and touching his toes","punching" wandb_mode="online" name=sd_cls_sds training.debug=False guidance.name=sd guidance.loss_type="classifier" model.video=False model.num_frames=1 training.iters=1000 training.accumulate=True training.constraint_latent_weight=0.1,1.0,2.0 model.initialize_pose="zero" model.model_change=False data.h=256 data.w=256 model.use_cubemap=True fp16=True training.scheduler=False model.add_fake_movement=True training.set_global_time_step=True training.lr=0.01,0.05,0.03,0.1 negative=""

# CUDA_VISIBLE_DEVICES=0 python -m  apps.run --multirun text="man" action="running","sitting" wandb_mode="online" name=first_video_sine training.debug=False hydra.verbose=False guidance.name=zeroscope,modelscope guidance.loss_type="standard" model.video=True model.num_frames=16,24 training.iters=1000 training.accumulate=True,False training.constraint_latent_weight=0.0 model.initialize_pose="zero" model.model_change=True,False data.h=256 data.w=256 model.use_cubemap=True fp16=True training.scheduler=True model.add_fake_movement=False,True training.set_global_time_step=True training.lr=0.05 training.guidance_scale=100 model.vpose=False

CUDA_VISIBLE_DEVICES=0 python -m apps.run --multirun \
    text="man" \
    action="running" \
    wandb_mode="online" \
    name=angle_optim_test \
    training.debug=False \
    hydra.verbose=False \
    guidance.name=no_guidance \
    guidance.loss_type="standard" \
    model.video=True \
    model.num_frames=20 \
    training.iters=2000 \
    training.accumulate=True \
    training.constraint_latent_weight=0.0 \
    training.optim="adam","adan" \
    model.initialize_pose="running_first_frame" \
    model.model_change=True \
    data.h=256 \
    data.w=256 \
    model.use_cubemap=False \
    fp16=True \
    training.scheduler="cosine","cosine_single","lambda","none" \
    model.add_fake_movement=False \
    training.set_global_time_step=False \
    training.lr=0.001,0.01 \
    training.guidance_scale=100 \
    model.vpose=False \
    model.pose_mlp_args.tau_scale=0.0,0.35,0.7,1.0 \
    model.pose_mlp_args.use_tanh_clamp=True \
    model.pose_mlp_args.tanh_scale=0.5,1.0 \
    training.use_ground_truth=True \
    model.pose_mlp="angle" \
