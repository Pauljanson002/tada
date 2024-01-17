use_6d = [True, False]

for u_6d in use_6d:
    name = f"six_d_use_6d={u_6d}"
    script = f'CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name={name} model.use_6d={u_6d}'
    print(script)
    # Execute the script or save it for later use
