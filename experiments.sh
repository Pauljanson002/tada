CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name=full_pose__model_change_True_full_pose_True model.model_change=True model.use_full_pose=True
CUDA_VISIBLE_DEVICES=1 python -m apps.run text="man" action="running" wandb_mode="online" name=full_pose__model_change_True_full_pose_False model.model_change=True model.use_full_pose=False
CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name=full_pose__model_change_False_full_pose_True model.model_change=False model.use_full_pose=True
CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name=full_pose__model_change_False_full_pose_False model.model_change=False model.use_full_pose=False
