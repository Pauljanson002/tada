CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="disabled" name=six_d_use_6d_True_model_change_False model.use_6d=True model.model_change=False training.debug=True
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "sitting" 
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "walking" 
