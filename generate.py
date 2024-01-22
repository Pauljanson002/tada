model_changes = [True, False]
use_full_pose = [True, False]
actions = ["running", "sitting","jumping"]
for model_change in model_changes:
    for full_pose in use_full_pose:
        name = f"full_pose__model_change_{model_change}_full_pose_{full_pose}"
        script = f'CUDA_VISIBLE_DEVICES=0 python -m apps.run text="man" action="running" wandb_mode="online" name={name} model.model_change={model_change} model.use_full_pose={full_pose}'
        print(script)
    # Execute the script or save it for later use
