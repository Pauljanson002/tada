use_6d = [True, False]
model_changes = [True, False]
for u_6d in use_6d:
    for model_change in model_changes:
        name = f"six_d_use_6d_{u_6d}_model_change_{model_change}"
        script = f'CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name={name} model.use_6d={u_6d} model.model_change={model_change}'
        print(script)
    # Execute the script or save it for later use
