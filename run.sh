CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" training.rgb_sds=True wandb_mode="disabled" name=test model.use_6d=False
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "sitting" 
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "walking" 
