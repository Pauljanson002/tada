CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action=running wandb_mode="online" name=action__model_change_True_full_pose_False_action_running model.model_change=True model.use_full_pose=False
CUDA_VISIBLE_DEVICES=1 python -m apps.run text="man" action=sitting wandb_mode="online" name=action__model_change_True_full_pose_False_action_sitting model.model_change=True model.use_full_pose=False
CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action=jumping wandb_mode="online" name=action__model_change_True_full_pose_False_action_jumping model.model_change=True model.use_full_pose=False
