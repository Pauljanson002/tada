CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name=six_d_use_6d=True model.use_6d=True
CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name=six_d_use_6d=False model.use_6d=False
