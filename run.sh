CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="disabled" model.model_change=True model.use_full_pose=True name=test training.debug=True
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "sitting" 
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "walking" 
