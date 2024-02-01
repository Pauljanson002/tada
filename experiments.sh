CUDA_VISIBLE_DEVICES=1 python -m apps.run --multirun text="man" action="running" wandb_mode="disabled" name=reg_acc_t training.debug=False guidance.name=zeroscope model.video=True model.num_frames=14 training.iters=1000 training.accumulate=True training.regularize_coeff=1000,100,10,1,0.1
CUDA_VISIBLE_DEVICES=2 python -m apps.run --multirun text="man" action="running" wandb_mode="disabled" name=reg_acc_f training.debug=False guidance.name=zeroscope model.video=True model.num_frames=14 training.iters=1000 training.accumulate=False training.regularize_coeff=1000,100,10,1,0.1

