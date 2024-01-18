CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name=six_d_use_6d_True_model_change_True model.use_6d=True model.model_change=True
CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name=six_d_use_6d_True_model_change_False model.use_6d=True model.model_change=False
CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name=six_d_use_6d_False_model_change_False model.use_6d=False model.model_change=False
