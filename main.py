import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.provider import ViewDataset
from lib.trainer import *
from lib.dlmesh import DLMesh
from lib.common.utils import load_config
from lib.schedulers import CosineAnnealingWarmupRestarts
import hydra
from omegaconf import OmegaConf

torch.autograd.set_detect_anomaly(False)


@hydra.main(version_base=None, config_path="configs", config_name="tada_wo_dpt.yaml")
def main(cfg):
    # cfg = argparse.Namespace(**OmegaConf.to_container(cfg))
    # cfg.freeze()
    hydra_singleton = hydra.core.hydra_config.HydraConfig.get()
    if str(hydra_singleton.mode) == "RunMode.MULTIRUN":
        if "SLURM_JOB_ID" in os.environ:
            job_id = hydra_singleton.job.id.split("_")[-1]
            cfg.name = f"{cfg.name}_{job_id}"
            cfg.training.workspace = os.path.join(hydra_singleton.sweep.dir, job_id)
        else:
            cfg.name = f"{cfg.name}_{hydra_singleton.job.id}"
            cfg.training.workspace = os.path.join(
                hydra_singleton.sweep.dir, hydra_singleton.job.id
            )
    else:
        cfg.training.workspace = hydra_singleton.run.dir

    seed_everything(cfg.seed)

    model = DLMesh(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.test:
        trainer = Trainer(
            cfg.name,
            text=cfg.text,
            action=cfg.action,
            negative=cfg.negative,
            dir_text=cfg.data.dir_text,
            opt=cfg.training,
            model=model,
            guidance=None,
            device=device,
            fp16=cfg.fp16,
        )

        test_loader = build_dataloader("test")
        trainer.test(test_loader)

        if cfg.save_mesh:
            trainer.save_mesh()

    else:
        train_loader = build_dataloader("train", cfg.view_count)

        scheduler, optimizer = configure_optimizer(cfg)
        try:
            guidance = configure_guidance(cfg)
            if isinstance(guidance, tuple):
                guidance, guidance_2 = guidance
            else:
                guidance_2 = None
        except:
            guidance = configure_guidance(cfg)
        wandb.init(
            project="tada",
            name=cfg.name,
            config=OmegaConf.to_container(cfg),
            tags=["phase_1"],
            mode=cfg.wandb_mode,
            reinit=True,
            group="_".join(cfg.name.split("_")[:-1]),
        )

        trainer = Trainer(
            cfg.name,
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
            scheduler_update_every_step=False,
            guidance_2=guidance_2,
        )
        if os.path.exists(cfg.data.image):
            trainer.default_view_data = train_loader.dataset.get_default_view_data()

        valid_loader = build_dataloader("val")
        max_epoch = np.ceil(
            cfg.training.iters / (len(train_loader) * train_loader.batch_size)
        ).astype(np.int32)
        print(f"max_epoch:{max_epoch}")

        trainer.train(train_loader, valid_loader, max_epoch)

        # test
        test_loader = build_dataloader("test")
        trainer.test(test_loader)
        if cfg.save_mesh:
            trainer.save_mesh()


def build_dataloader(cfg, phase, view_count=4):
        """
        Args:
            phase: str one of ['train', 'test' 'val']
        Returns:
        """
        size = 100 if phase == "val" else view_count
        dataset = ViewDataset(cfg.data, device=cfg.device, type=phase, size=size)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

def configure_guidance(cfg):
    opt = cfg.guidance

    if "both" in opt.name:
        from lib.guidance.sd import StableDiffusion
        from lib.guidance.zeroscope import ZeroScope
        from lib.guidance.video_crafter import VideoCrafter
        from lib.guidance.modelscope import ModelScope

        # return StableDiffusion(device, cfg.fp16, opt.vram_O,t_range=[opt.t_start,opt.t_end],loss_type=opt.loss_type), ZeroScope(device,cfg.fp16, opt.vram_O,t_range=[opt.t_start,opt.t_end],loss_type=None)
        if opt.name == "both_vc":
            return StableDiffusion(
                cfg.device,
                cfg.fp16,
                opt.vram_O,
                t_range=[opt.t_start, opt.t_end],
                loss_type=opt.loss_type,
            ), VideoCrafter(
                cfg.device,
                cfg.fp16,
                opt.vram_O,
                t_range=[opt.t_start, opt.t_end],
                loss_type=None,
            )
        elif opt.name == "both_ms":
            return StableDiffusion(
                cfg.device,
                cfg.fp16,
                opt.vram_O,
                t_range=[opt.t_start, opt.t_end],
                loss_type=opt.loss_type,
            ), ModelScope(
                cfg.device,
                cfg.fp16,
                opt.vram_O,
                t_range=[opt.t_start, opt.t_end],
                loss_type=None,
            )
        elif opt.name == "both_zs":
            return StableDiffusion(
                cfg.device,
                cfg.fp16,
                opt.vram_O,
                t_range=[opt.t_start, opt.t_end],
                loss_type=opt.loss_type,
            ), ZeroScope(
                cfg.device,
                cfg.fp16,
                opt.vram_O,
                t_range=[opt.t_start, opt.t_end],
                loss_type=None,
            )
        else:
            raise NotImplementedError()
    elif opt.name == "sd":
        from lib.guidance.sd import StableDiffusion

        return StableDiffusion(
            cfg.device,
            cfg.fp16,
            opt.vram_O,
            opt.sd_version,
            t_range=[opt.t_start, opt.t_end],
            loss_type=opt.loss_type,
        )
    elif opt.name == "if":
        from lib.guidance.deepfloyd import IF

        return IF(cfg.device, opt.vram_O)
    elif opt.name == "zeroscope":
        from lib.guidance.zeroscope import ZeroScope

        return ZeroScope(
            cfg.device,
            cfg.fp16,
            opt.vram_O,
            t_range=[opt.t_start, opt.t_end],
            loss_type=None,
        )
    elif opt.name == "no_guidance":
        from lib.guidance.no_guidance import NoGuidance

        return NoGuidance()
    elif opt.name == "naive":
        from lib.guidance.naive_guidance import Naive

        return Naive()
    elif opt.name == "modelscope":
        from lib.guidance.modelscope import ModelScope

        return ModelScope(
            cfg.device,
            cfg.fp16,
            opt.vram_O,
            t_range=[opt.t_start, opt.t_end],
            loss_type=None,
        )
    elif opt.name == "videocrafter":
        from lib.guidance.video_crafter import VideoCrafter

        return VideoCrafter(
            cfg.device,
            cfg.fp16,
            opt.vram_O,
            t_range=[opt.t_start, opt.t_end],
            loss_type=None,
        )
    else:
        from lib.guidance.clip import CLIP

        return CLIP(cfg.device)

def configure_optimizer(cfg):
    opt = cfg.training
    if opt.optim == "adan":
        from lib.common.optimizer import Adan

        optimizer = lambda model: Adan(
            model.get_params(opt.lr * 10),
            eps=1e-8,
            weight_decay=2e-5,
            max_grad_norm=5.0,
            foreach=False,
        )
    else:  # adam
        optimizer = lambda model: torch.optim.Adam(
            model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15
        )

    if opt.scheduler == "cosine":
        scheduler = lambda optimizer: CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(opt.iters * 0.25 * 0.2),
            cycle_mult=1.0,
            max_lr=opt.lr,
            min_lr=opt.lr / 10,
            warmup_steps=int(opt.iters * 0.25 * 0.05),
            gamma=1.0,
        )
    elif opt.scheduler == "cosine_single":
        scheduler = lambda optimizer: CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(opt.iters),
            cycle_mult=1.0,
            max_lr=opt.lr,
            min_lr=opt.lr / 10,
            warmup_steps=int(opt.iters * 0.25 * 0.1),
            gamma=1.0,
        )
    elif opt.scheduler == "lambda":
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda x: 0.1 ** min(x / opt.iters, 1)
        )
    else:
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda x: 1 ** min(x / opt.iters, 1)
        )
    return scheduler, optimizer


if __name__ == "__main__":
    main()
