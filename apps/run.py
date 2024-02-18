import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.provider import ViewDataset
from lib.trainer import *
from lib.dlmesh import DLMesh
from lib.common.utils import load_config
import hydra
from omegaconf import OmegaConf
torch.autograd.set_detect_anomaly(False)


@hydra.main(version_base=None,config_path="../configs", config_name="tada_wo_dpt.yaml")
def main(cfg):
    # cfg = argparse.Namespace(**OmegaConf.to_container(cfg))
    # cfg.freeze()
    hydra_singleton = hydra.core.hydra_config.HydraConfig.get()
    if str(hydra_singleton.mode) == "RunMode.MULTIRUN":
        if "SLURM_JOB_ID" in os.environ:
            job_id =  hydra_singleton.job.id.split("_")[-1]
            cfg.name = f"{cfg.name}_{job_id}"
            cfg.training.workspace = os.path.join(hydra_singleton.sweep.dir,job_id)
        else:
            cfg.name = f"{cfg.name}_{hydra_singleton.job.id}"
            cfg.training.workspace = os.path.join(hydra_singleton.sweep.dir,hydra_singleton.job.id)
    else:
        cfg.training.workspace  = hydra_singleton.run.dir
    # save config to workspace
    if False: #disabling this for now
        os.makedirs(os.path.join(cfg.training.workspace, cfg.text,cfg.action), exist_ok=True)
        with open(os.path.join(cfg.training.workspace,cfg.text,cfg.action,"config.yaml"), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    def build_dataloader(phase):
        """
        Args:
            phase: str one of ['train', 'test' 'val']
        Returns:
        """
        size = 100 if phase == 'val' else 4
        dataset = ViewDataset(cfg.data, device=device, type=phase, size=size)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def configure_guidance():
        opt = cfg.guidance
        if opt.name == 'sd':
            from lib.guidance.sd import StableDiffusion
            return StableDiffusion(device, cfg.fp16, opt.vram_O, opt.sd_version)
        elif opt.name == 'if':
            from lib.guidance.deepfloyd import IF
            return IF(device, opt.vram_O)
        elif opt.name == "zeroscope":
            from lib.guidance.zeroscope import ZeroScope
            return ZeroScope(device,cfg.fp16, opt.vram_O,t_range=[opt.t_start,opt.t_end])
        elif opt.name == "no_guidance":
            from lib.guidance.no_guidance import NoGuidance
            return NoGuidance()
        elif opt.name == "naive":
            from lib.guidance.naive_guidance import Naive
            return Naive()
        elif opt.name == "modelscope":
            from lib.guidance.modelscope import ModelScope
            return ModelScope(device, cfg.fp16, opt.vram_O,t_range=[opt.t_start,opt.t_end])
        else:
            from lib.guidance.clip import CLIP
            return CLIP(device)

    def configure_optimizer():
        opt = cfg.training
        if opt.optim == 'adan':
            from lib.common.optimizer import Adan

            optimizer = lambda model: Adan(
                model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else:  # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(5 * opt.lr), betas=(0.9, 0.99), eps=1e-15)

        if opt.scheduler == True:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.1 ** min(x / opt.iters, 1))
        else:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1 ** min(x / opt.iters, 1))
        return scheduler, optimizer

    model = DLMesh(cfg.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.test:
        trainer = Trainer(cfg.name,
                          text=cfg.text,
                          action=cfg.action,
                          negative=cfg.negative,
                          dir_text=cfg.data.dir_text,
                          opt=cfg.training,
                          model=model,
                          guidance=None,
                          device=device,
                          fp16=cfg.fp16
                          )

        test_loader = build_dataloader('test')
        trainer.test(test_loader)

        if cfg.save_mesh:
            trainer.save_mesh()

    else:
        train_loader = build_dataloader('train')

        scheduler, optimizer = configure_optimizer()
        try:
            guidance = configure_guidance()
        except:
            guidance = configure_guidance()
        wandb.init(project="tada",name=cfg.name,config=OmegaConf.to_container(cfg),tags=["tada"],mode=cfg.wandb_mode,reinit=True)


        trainer = Trainer(cfg.name,
                        text=cfg.text,
                        action=cfg.action,
                        negative=cfg.negative,
                        dir_text=cfg.data.dir_text,
                        opt=cfg.training,
                        model=model,
                        guidance=guidance,
                        device=device,
                        optimizer=optimizer,
                        fp16=cfg.fp16,
                        lr_scheduler=scheduler,
                        scheduler_update_every_step=True
                        )
        if os.path.exists(cfg.data.image):
            trainer.default_view_data = train_loader.dataset.get_default_view_data()

        valid_loader = build_dataloader('val')
        max_epoch = np.ceil(cfg.training.iters / (len(train_loader) * train_loader.batch_size)).astype(np.int32)
        print(f"max_epoch:{max_epoch}")
        trainer.train(train_loader, valid_loader, max_epoch)

        # test
        test_loader = build_dataloader('test')
        trainer.test(test_loader)
        if cfg.save_mesh:
            trainer.save_mesh()

if __name__ == '__main__':
    main()