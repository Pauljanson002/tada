rgb_sds = [True, False]
normal_sds = [True, False]
mean_sds = [True, False]

for rgb in rgb_sds:
    for normal in normal_sds:
        for mean in mean_sds:
            name = f"running_rgb_sds_{rgb}_normal_sds_{normal}_mean_sds_{mean}"
            script = f'CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" training.rgb_sds={rgb} training.normal_sds={normal} training.mean_sds={mean} wandb_mode="online" name={name}'
            print(script)
            # Execute the script or save it for later use
