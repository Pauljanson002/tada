UDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" training.rgb_sds=True training.normal_sds=True training.mean_sds=True wandb_mode="disabled" name=running_rgb_sds_True_normal_sds_True_mean_sds_True
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "sitting" 
# CUDA_VISIBLE_DEVICES=0 python -m apps.run --config configs/tada_wo_dpt.yaml --text "man" --action "walking" 
