import os
import submitit

def main():
    os.system('python -m apps.run text="man" action="running" wandb_mode="disabled" name=six_d_use_6d_True_model_change_False model.use_6d=True model.model_change=False training.debug=True')
    print("Done")
    return True
    

log_folder = "/scratch/janson2/tmp"
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(timeout_min=60,slurm_account="rrg-eugenium",gpus_per_node=1)

job = executor.submit(main) 

print(job.job_id)

output = job.result()
print(output)