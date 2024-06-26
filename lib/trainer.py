import glob
import pickle
import random
import tqdm
import imageio
import tensorboardX
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from pytorch3d.ops.knn import knn_points
# from pytorch3d.loss import chamfer_distance
import torch.distributed as dist
from rich.console import Console
from torch_ema import ExponentialMovingAverage
import wandb
from lib.guidance.no_guidance import NoGuidance
from lib.guidance.naive_guidance import Naive
from lib.common.utils import *
from lib.dpt import DepthNormalEstimation
from lib.rotation_conversions import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)
import wandb
import logging


class Trainer(object):
    def __init__(
        self,
        name,  # name of this experiment
        text,
        action,
        negative,
        dir_text,
        opt,  # extra conf
        model,  # network
        guidance,  # guidance network
        guidance_2 = None,
        criterion=None,  # loss function, if None, assume inline implementation in step
        optimizer=None,  # optimizer
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        metrics=[],
        # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        local_rank=0,  # which GPU am I
        world_size=1,  # total num of GPUs
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,  # whether to mute all print
        fp16=False,  # amp optimize level
        max_keep_ckpt=2,  # max num of saved ckpts in disk
        best_mode="min",  # the smaller/larger result, the better
        use_loss_as_metric=True,  # use loss as the first metric
        report_metric_at_train=False,  # also report metrics at training
        use_tensorboardX=True,  # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
    ):
        self.dpt = DepthNormalEstimation(use_depth=False) if opt.use_dpt else None
        self.default_view_data = None
        self.name = name
        self.text = text
        self.action = action
        self.negative = negative
        self.dir_text = dir_text
        self.context = "beach"
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = os.path.join(opt.workspace)
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = opt.eval_interval
        self.use_checkpoint = opt.ckpt
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (
            device
            if device is not None
            else torch.device(
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )
        )
        self.console = Console()

        self.model = model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2, 3]).module
        if self.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank]
            )

        # guide model
        self.guidance = guidance
        self.guidance_2 = guidance_2
        # self.guidance = torch.nn.DataParallel(self.guidance, device_ids=[0, 1, 2, 3]).module
        # text prompt

        self.running_body_pose = pickle.load(open("4d/poses/running.pkl", "rb"))[
            "body_pose"
        ]
        self.running_body_pose = (
            torch.as_tensor(self.running_body_pose).to("cuda").float()
        )
        self.running_body_pose = self.running_body_pose[: self.model.opt.num_frames, :]
        self.running_body_pose = matrix_to_rotation_6d(
            axis_angle_to_matrix(self.running_body_pose.view(-1, 3))
        ).view(self.model.opt.num_frames, -1)

        self.landmarks = []
        for i in range(4):
            ldnmrk = torch.load(f"smplx_joints_{i}.pt").to("cuda")
            ldnmrk.requires_grad = False
            self.landmarks.append(ldnmrk)
        self.landmarks_3d = torch.load("smplx_3d_joints.pt").to("cuda")
        self.text_embeds = None
        if self.guidance is not None:
            for p in self.guidance.parameters():
                p.requires_grad = False
            self.prepare_text_embeddings()
        if self.guidance_2 is not None:
            for p in self.guidance_2.parameters():
                p.requires_grad = False

        # # try out torch 2.0
        # if torch.__version__[0] == '2':
        #     self.model = torch.compile(self.model)
        #     self.guidance = torch.compile(self.guidance)

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=0.001, weight_decay=5e-4
            )  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=ema_decay
            )
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = "min"

        # workspace prepare
        self.logger = logging.getLogger(__name__)
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(self.workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")
            if True:
                self.ckpt_path = os.path.join(self.workspace, "checkpoints")
                self.best_path = f"{self.ckpt_path}/{self.name}.pth"
                os.makedirs(self.ckpt_path, exist_ok=True)

        self.logger.info(
            f' Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.logger.info(
            f" #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}"
        )
        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.logger.info(" Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.logger.info(" Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.logger.info(" Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.logger.info(" Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.logger.info(f" {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.logger.info(f" Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint, model_only=True)

        self.train_video_frames = []
        self.write_train_video = True

        if self.opt.landmarks_count > 0:
            self.random_joint_mask = torch.randint(0, 70, (self.opt.landmarks_count,))
        else:
            self.opt.use_landmarks = False

        self.save_freq = 50

    # calculate the text embeddings.
    # Show : Text embeddings
    def prepare_text_embeddings(self):
        if self.text is None:
            self.logger.warning(f" text prompt is not provided.")
            return

        if self.action is not None:
            self.text_embeds = {
                "uncond": self.guidance.get_text_embeds([self.negative]),
                "default": self.guidance.get_text_embeds(
                    [
                        f"a shot of a {self.text} {self.action} , full-body"
                    ]
                ),
            }
            self.text_embeds_ref = {
                "uncond": self.guidance.get_text_embeds([self.negative]),
                "default": self.guidance.get_text_embeds(
                    [f"a shot of a {self.text} in the , full-body"]
                ),
            }
        else:
            self.text_embeds = {
                "uncond": self.guidance.get_text_embeds([self.negative]),
                "default": self.guidance.get_text_embeds(
                    [f"a shot of a {self.subject}"]
                ),
            }

        if self.opt.train_face_ratio < 1:
            if self.action is not None:
                self.text_embeds["body"] = {
                    d: self.guidance.get_text_embeds(
                        [
                            f"a shot of {d} view of a {self.text} {self.action} , full-body"
                        ]
                    )
                    for d in ["front", "side", "back", "overhead"]
                }
                self.text_embeds_ref["body"] = {
                    d: self.guidance.get_text_embeds(
                        [
                            f"a shot of {d} view of a {self.text} , full-body"
                        ]
                    )
                    for d in ["front", "side", "back", "overhead"]
                }
            else:
                self.text_embeds["body"] = {
                    d: self.guidance.get_text_embeds(
                        [
                            f"a shot of {d} view of a {self.text} , full-body"
                        ]
                    )
                    for d in ["front", "side", "back", "overhead"]
                }
        if self.opt.train_face_ratio > 0:
            id_text = self.text.split("wearing")[0]
            self.text_embeds["face"] = {
                d: self.guidance.get_text_embeds(
                    [f"a {d} view 3D rendering of {id_text}, face"]
                )
                for d in ["front", "side", "back"]
            }

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    # def log(self, *args, **kwargs):
    #     if self.local_rank == 0:
    #         if not self.mute:
    #             # print(*args)
    #             self.console.print(*args, **kwargs)
    #         if self.log_ptr:
    #             print(*args, file=self.log_ptr)
    #             self.log_ptr.flush()  # write immediately to file

    def train_step(self, data, is_full_body, **kwargs):
        do_rgbd_loss = self.default_view_data is not None and (
            self.global_step % self.opt.known_view_interval == 0
        )
        loss_dict = {}
        if do_rgbd_loss:
            data = self.default_view_data

        H, W = data["H"], data["W"]
        mvp = data["mvp"]  # [B, 4, 4]
        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]

        # TEST: progressive training resolution
        if self.opt.anneal_tex_reso:
            scale = min(1, self.global_step / (0.8 * self.opt.iters))

            def make_divisible(x, y):
                return x + (y - x % y)

            H = max(make_divisible(int(H * scale), 16), 32)
            W = max(make_divisible(int(W * scale), 16), 32)

        if do_rgbd_loss and self.opt.known_view_noise_scale > 0:
            noise_scale = (
                self.opt.known_view_noise_scale
            )  # * (1 - self.global_step / self.opt.iters)
            rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
            rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================

        dir_text_z = [
            self.text_embeds["uncond"],
            self.text_embeds[data["camera_type"][0]][data["dirkey"][0]],
        ]
        dir_text_z_ref = [
            self.text_embeds_ref["uncond"],
            self.text_embeds_ref[data["camera_type"][0]][data["dirkey"][0]],
        ]
        dir_text_z = torch.cat(dir_text_z)
        dir_text_z_ref = torch.cat(dir_text_z_ref)
        out = self.model(rays_o, rays_d, mvp, data["H"], data["W"], shading=self.model.shading)
        video = self.model.opt.video
        if not video:
            image = out["image"].permute(0, 3, 1, 2)
            normal = out["normal"].permute(0, 3, 1, 2)
            alpha = out["alpha"].permute(0, 3, 1, 2)

        # Show losses
        if not video:
            out_annel = self.model(rays_o, rays_d, mvp, H, W, shading=self.model.shading)
            image_annel = out_annel["image"].permute(0, 3, 1, 2)
            normal_annel = out_annel["normal"].permute(0, 3, 1, 2)
            alpha_annel = out_annel["alpha"].permute(0, 3, 1, 2)

            pred = torch.cat([out["image"], out["normal"]], dim=2)
            pred = (pred[0].detach().cpu().numpy() * 255).astype(np.uint8)

            p_iter = self.global_step / self.opt.iters

        else:

            video_frames = out["video"].squeeze(1)

            normal_frames = out["normal_vid"].squeeze(1)
            alpha_frames = out["alpha_vid"].squeeze(1)
            video_frames_np = (video_frames.detach().cpu().numpy() * 255).astype(
                np.uint8
            )
            video_frames = video_frames.permute(0, 3, 1, 2)

            normal_frames = normal_frames.permute(0, 3, 1, 2)
            alpha_frames = alpha_frames.permute(0, 3, 1, 2)
            pred = video_frames_np

            if self.opt.rgb_sds:
                selected_frames = torch.randint(0, video_frames.size(0), (self.model.num_frames//2,))
                loss = self.opt.g1_coeff * (
                    1.0
                    * self.guidance.train_step(
                        dir_text_z,
                        video_frames[selected_frames],
                        view_id=kwargs.get("view_id", 0),
                        guidance_scale=self.opt.guidance_scale,
                    ).mean()
                )
                loss_dict[f"individual_sds/{str(type(self.guidance))}"] = loss.item()
                self.scaler.scale(loss).backward()
                if self.guidance_2 is not None:
                    out = self.model(rays_o, rays_d, mvp, data["H"], data["W"], shading=self.model.shading)
                    video = self.model.opt.video
                    video_frames = out["video"].squeeze(1)
                    video_frames = video_frames.permute(0, 3, 1, 2)
                    loss = self.opt.g2_coeff * (
                        1.0
                        * self.guidance_2.train_step(
                            dir_text_z,
                            video_frames,
                            view_id=kwargs.get("view_id", 0),
                            guidance_scale=self.opt.guidance_scale,
                        ).mean()
                    )
                    loss_dict[f"individual_sds/{str(type(self.guidance_2))}"] = loss.item()

                loss_dict["rgb_sds"] = loss.item()
            elif self.opt.normal_sds:
                loss = self.guidance.train_step(
                    dir_text_z, normal_frames, guidance_scale=self.opt.guidance_scale
                ).mean()
                loss_dict["normal_sds"] = loss.item()
            elif self.opt.mean_sds:
                loss = self.guidance.train_step(
                    dir_text_z, torch.cat([normal_frames, video_frames.detach()])
                ).mean()
                loss_dict["mean_sds"] = loss.item()
            else:
                loss = 0

            if self.opt.constraint_latent_weight > 0 and self.model.vpose:
                if self.model.opt.pose_mlp is not None:
                    constraint_loss = (
                        self.opt.constraint_latent_weight
                        * torch.norm(out["prediction"]).mean()
                    )
                    loss += constraint_loss
                    loss_dict["constraint_latent"] = constraint_loss.item()
                else:
                    constraint_loss = (
                        self.opt.constraint_latent_weight
                        * torch.norm(self.model.body_pose_6d_set, dim=1).mean()
                    )
                    loss += constraint_loss
                    loss_dict["constraint_latent"] = constraint_loss.item()

            if self.model.opt.pose_mlp is not None:
                encoded_vpose = self.model.body_prior.encode(
                    matrix_to_axis_angle(
                        rotation_6d_to_matrix((out["prediction"]+self.model.init_body_pose_6d_set).view(-1, 6))
                    ).view(self.model.num_frames, -1)
                ).mean
                # add quadratic penalty to the latent space
                loss += self.opt.q_p**2 * encoded_vpose.pow(2).sum()

            if self.opt.use_ground_truth:
                # L2 loss between the body pose 6d and the running pose
                loss += F.mse_loss(
                    out["prediction"].view(-1, 6), self.running_body_pose.view(-1, 6)
                )
                # loss += F.mse_loss(out["prediction"], torch.zeros_like(out["prediction"]))
                self.logger.debug(f"Ground truth loss: {loss.item()}")
                wandb.log(
                    {
                        "loss/ground_truth_loss": loss.item(),
                        "epoch": self.epoch,
                    }
                )
            else:
                # L2 loss between the body pose 6d and the running pose
                if self.model.vpose:
                    dummy_loss = F.mse_loss(
                        matrix_to_rotation_6d(
                            axis_angle_to_matrix(
                                self.model.body_prior.decode(
                                    out["prediction"].unsqueeze(0)
                                )["pose_body"]
                                .contiguous()
                                .view(-1, 3)
                            )
                        ).view(self.model.num_frames, -1),
                        self.running_body_pose,
                    )
                else:
                    if self.model.opt.pose_mlp != "none":
                        dummy_loss = F.mse_loss(
                            out["prediction"].view(-1, 6),
                            self.running_body_pose.view(-1, 6),
                        )
                        self.logger.debug(f"Dummy loss: {dummy_loss.item()}")
                        wandb.log(
                            {
                                "loss/dummy_truth_loss": dummy_loss.item(),
                                "epoch": self.epoch,
                            }
                        )
                        del dummy_loss
            # if self.model.vpose:
            #     # Constraint the size of the body pose 6d to norm 1
            #     loss += torch.norm(self.model.body_pose_6d_set, dim=1).mean()

            if self.opt.use_landmarks:
                # L2 loss between the landmarks and the predicted landmarks
                landmark_2d_loss = 1 * F.mse_loss(
                    (
                        out["smplx_joints_vid"].view(self.model.num_frames, -1, 2)[
                            :, self.random_joint_mask, :
                        ]
                    ).view(-1, 2),
                    (
                        self.landmarks[self.local_step - 1].view(
                            self.model.num_frames, -1, 2
                        )[:, self.random_joint_mask, :]
                    ).view(-1, 2),
                )
                self.logger.debug(
                    f"Landmark loss {self.local_step}: {landmark_2d_loss.item()}"
                )
                loss += landmark_2d_loss
                wandb.log(
                    {
                        "loss/landmark_loss": landmark_2d_loss.item(),
                        "epoch": self.epoch,
                    }
                )

            if self.opt.use_3d_landmarks:
                # L2 loss between the landmarks and the predicted landmarks
                landmark_3d_loss = 1 * F.mse_loss(
                    out["smplx_3d_joints_vid"].view(-1, 3),
                    self.landmarks_3d.view(-1, 3),
                )
                loss += landmark_3d_loss
                self.logger.debug(f"3D Landmark loss: {landmark_3d_loss.item()}")
                wandb.log(
                    {
                        "loss/3d_landmark_loss": landmark_3d_loss.item(),
                        "epoch": self.epoch,
                    }
                )

            if self.opt.regularize_coeff > 0:
                if self.model.opt.pose_mlp is not None:
                    difference = out["prediction"][1:] - out["prediction"][:-1]
                else:
                    difference = (
                        self.model.body_pose_6d_set[1:]
                        - self.model.body_pose_6d_set[:-1]
                    )
                regularization_term = torch.sum(difference * difference)
                loss += self.opt.regularize_coeff * regularization_term
                loss_dict["reg_loss"] = regularization_term.item()
            else:
                loss_dict["reg_loss"] = 0

            # TODO: Implement normal sds
            # if self.opt.normal_sds:
            #     loss += self.guidance.train_step(dir_text_z, normal_frames).mean()
            # if self.opt.mean_sds:
            #     loss += self.guidance.train_step(dir_text_z, torch.cat([normal_frames, video_frames.detach()])).mean() #TODO: Why detach video frames?

        if do_rgbd_loss:  # with image input
            # gt_mask = data['mask']  # [B, H, W]
            gt_rgb = data["rgb"]  # [B, 3, H, W]
            gt_normal = data["normal"]  # [B, H, W, 3]
            gt_depth = data["depth"]  # [B, H, W]
            # rgb loss
            loss = self.opt.lambda_rgb * F.mse_loss(image, gt_rgb)
            # normal loss
            if self.opt.lambda_normal > 0:
                lambda_normal = self.opt.lambda_normal * min(
                    1, self.global_step / self.opt.iters
                )
                loss = loss + lambda_normal * (
                    1 - F.cosine_similarity(normal, gt_normal).mean()
                )
            # depth loss
            if self.opt.lambda_depth > 0:
                depth = None
                lambda_depth = self.opt.lambda_depth * min(
                    1, self.global_step / self.opt.iters
                )
                loss = loss + lambda_depth * (1 - self.pearson(depth, gt_depth))
        else:
            # rgb sds
            if not video:
                if self.opt.rgb_sds:
                    if self.opt.dds:
                        loss = self.guidance.train_step(dir_text_z,image_annel,dds_embeds=dir_text_z_ref).mean()
                        loss_dict["rgb_sds"] = loss.item()
                    else:
                        loss = self.guidance.train_step(dir_text_z, image_annel).mean()
                        loss_dict["rgb_sds"] = loss.item()
                else:
                    loss = 0

                if self.opt.constraint_latent_weight > 0 and self.model.vpose:
                    constraint_loss = (
                        self.opt.constraint_latent_weight
                        * torch.norm(self.model.body_pose_6d, dim=1).mean()
                    )
                    loss += constraint_loss
                    loss_dict["constraint_latent"] = constraint_loss.item()

                if not self.dpt:
                    # normal sds
                    if self.opt.normal_sds:
                        loss += self.guidance.train_step(dir_text_z, normal).mean()
                    # latent mean sds
                    if self.opt.mean_sds:
                        loss += self.guidance.train_step(
                            dir_text_z, torch.cat([normal, image.detach()])
                        ).mean()
                else:
                    if p_iter < 0.3 or random.random() < 0.5:
                        # normal sds
                        loss += self.guidance.train_step(dir_text_z, normal).mean()
                    elif self.dpt is not None:
                        # normal image loss
                        dpt_normal = self.dpt(image)
                        dpt_normal = (1 - dpt_normal) * alpha + (1 - alpha)
                        lambda_normal = self.opt.lambda_normal * min(
                            1, self.global_step / self.opt.iters
                        )
                        loss += lambda_normal * (
                            1 - F.cosine_similarity(normal, dpt_normal).mean()
                        )

                        # pred = np.hstack([(normal[0]).permute(1, 2, 0).detach().cpu().numpy(),
                        #                   dpt_normal[0].permute(1, 2, 0).detach().cpu().numpy()]) * 255
                        # cv2.imwrite("im.png", pred)
                        # exit()

        return pred, loss, loss_dict

    def eval_step(self, data):
        H, W = data["H"].item(), data["W"].item()
        mvp = data["mvp"]
        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]
        out = self.model(rays_o, rays_d, mvp, H, W, shading="old", is_train=False)
        if not self.model.opt.video:
            w = out["normal"].shape[2]
            pred = torch.cat(
                [
                    out["normal"],
                    out["image"],
                    torch.cat(
                        [out["normal"][:, :, : w // 2], out["image"][:, :, w // 2 :]],
                        dim=2,
                    ),
                ],
                dim=1,
            )
        else:
            pred = out["video"].squeeze(1)

        # dummy
        loss = torch.zeros([1], device=self.device, dtype=torch.float32)

        return pred, loss

    def test_step(self, data):
        H, W = data["H"].item(), data["W"].item()
        mvp = data["mvp"]
        rays_o = data["rays_o"]  # [B, N, 3]
        rays_d = data["rays_d"]  # [B, N, 3]
        out = self.model(rays_o, rays_d, mvp, H, W, shading="old", is_train=False)

        if not self.model.opt.video:
            w = out["normal"].shape[2]
            pred = torch.cat(
                [
                    out["normal"],
                    out["image"],
                    torch.cat(
                        [out["normal"][:, :, : w // 2], out["image"][:, :, w // 2 :]],
                        dim=2,
                    ),
                ],
                dim=2,
            )
        else:
            pred = out["video"].squeeze(1)

        return pred, None

    def save_mesh(self, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, "mesh")

        self.logger.info(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path)

        self.logger.info(f"==> Finished saving mesh.")

    def train(self, train_loader, valid_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name)
            )

        save_path = os.path.join(self.workspace, "results")
        os.makedirs(save_path, exist_ok=True)

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            with torch.no_grad():
                #! Here they select the full body or face to train for this epoch
                if random.random() < self.opt.train_face_ratio:
                    train_loader.dataset.full_body = False
                    face_center, face_scale = self.model.get_mesh_center_scale("face")
                    train_loader.dataset.face_center = face_center
                    train_loader.dataset.face_scale = face_scale.item() * 10

                else:
                    train_loader.dataset.full_body = True
                    # body_center, body_scale = self.model.get_mesh_center_scale("body")
                    # train_loader.dataset.body_center = body_center
                    # train_loader.dataset.body_scale = body_scale.item()
            self.train_one_epoch(train_loader)

            if (
                self.workspace is not None and self.local_rank == 0
            ) and False:  # Disabling this for now
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=False)
                if self.write_train_video:
                    if self.model.opt.video:
                        all_preds = np.stack(self.train_video_frames, axis=0)
                    else:
                        all_preds = np.concatenate(self.train_video_frames, axis=0)
                    imageio.mimwrite(
                        os.path.join(self.workspace, "results", f"train_vis.mp4"),
                        all_preds,
                        fps=25,
                        quality=5,
                        macro_block_size=1,
                    )
            if self.opt.debug:
                break

        end_t = time.time()

        self.logger.info(f" training takes {(end_t - start_t) / 60:.4f} minutes.")
        if self.write_train_video:
            if self.model.opt.video:
                all_preds = np.stack(self.train_video_frames, axis=0)
            else:
                all_preds = np.concatenate(self.train_video_frames, axis=0)
            imageio.mimwrite(
                os.path.join(self.workspace, "results", f"train_vis.mp4"),
                all_preds,
                fps=25,
                quality=5,
                macro_block_size=1,
            )
            self.logger.info(f"==> Finished writing train video.")
            # try:
            #     wandb.log({
            #         "train_video": wandb.Video(os.path.join(self.workspace,"results", f'train_vis.mp4'), fps=25, format="mp4")
            #     })
            # except:
            #     pass

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, "results")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        os.makedirs(save_path, exist_ok=True)

        self.logger.info(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model.eval()

        if write_video:
            all_preds = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, _ = self.test_step(data)

                if not self.model.opt.video:
                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)
                else:
                    pred = preds.detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)
                    # pred = pred.transpose(1,0,2,3)
                    # pred = pred.reshape(pred.shape[0], -1, pred.shape[3])
                if write_video:
                    all_preds.append(pred)
                else:
                    os.makedirs(os.path.join(save_path, "image"), exist_ok=True)
                    cv2.imwrite(
                        os.path.join(save_path, "image", f"{i:04d}.png"),
                        cv2.cvtColor(pred[..., :3], cv2.COLOR_RGB2BGRA),
                    )

                pbar.update(loader.batch_size)

        if write_video:
            if not self.model.opt.video:
                all_preds = np.stack(all_preds, axis=0)

                imageio.mimwrite(
                    os.path.join(save_path, f"{name}.mp4"),
                    all_preds,
                    fps=25,
                    quality=9,
                    macro_block_size=1,
                )
            else:
                for i in range(len(all_preds)):
                    imageio.mimwrite(
                        os.path.join(save_path, f"{name}_view_{i}.mp4"),
                        all_preds[i],
                        fps=5,
                        quality=5,
                        macro_block_size=1,
                    )
                    # try:
                    #     wandb.log({
                    #         f"test_video_{i}": wandb.Video(os.path.join(save_path, f'{name}_view_{i}.mp4'), fps=5, format="mp4")
                    #     })
                    # except:
                    #     pass

        self.logger.info(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.logger.info(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )

        total_sds_loss = 0
        total_reg_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        video = self.model.opt.video
        self.local_step = 0
        temp_grads = {
            n_p: 0 for n_p, p in self.model.named_parameters() if p.requires_grad
        }
        if self.opt.set_global_time_step:
            self.guidance.global_time_step = torch.randint(
                self.guidance.min_step,
                self.guidance.max_step + 1,
                (1,),
                dtype=torch.long,
                device=self.device,
            )
        loss_list = []
        for view_id, data in enumerate(loader):

            # if view_id in [0,2]:
            #     print(f"Skipping view {view_id}: {data['dirkey'][0]}")
            #     continue

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            pred_rgbs, loss, loss_dict = self.train_step(
                data, loader.dataset.full_body, view_id=view_id
            )

            if self.global_step % self.save_freq == 0:
                if not video:
                    pred = cv2.cvtColor(pred_rgbs, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(
                        self.workspace,
                        "train-vis",
                        f"{self.name}/{self.global_step:04d}.png",
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, pred)
                else:
                    save_path = os.path.join(
                        self.workspace,
                        "train-vis",
                        f"{self.name}/{self.global_step:04d}.mp4",
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    imageio.mimwrite(
                        save_path, pred_rgbs, fps=3, quality=5, macro_block_size=1
                    )

            # Add the first frame for reference
            if self.global_step == 1 and self.model.opt.video:
                pred_np = (pred_rgbs).astype(np.uint8)
                t_pred = pred_np.transpose(1, 0, 2, 3)
                t_pred = t_pred.reshape(t_pred.shape[0], -1, t_pred.shape[3])
                self.train_video_frames.append(t_pred)
            try:
                self.scaler.scale(loss).backward()
            except RuntimeError as e:
                print(e)
            if self.opt.accumulate:
                for n_p, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        temp_grads[n_p] += p.grad

            if not self.opt.accumulate:
                try:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler_update_every_step:
                        self.lr_scheduler.step()
                except Exception as e:
                    print(e)

            loss_val = loss.item()
            total_sds_loss += loss_dict.get("rgb_sds", 0)
            total_reg_loss += loss_dict.get("reg_loss", 0)
            wandb.log(
                {
                    **{"loss/" + k: v for k, v in loss_dict.items()},
                    "local_step": self.local_step,
                    "epoch": self.epoch,
                }
            )

            if self.local_rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )
                if self.scheduler_update_every_step:
                    pbar.set_description(
                        "loss={:.4f} , Reg loss{:.4f}, lr={:.6f}, ".format(
                            loss_val,
                            loss_dict.get("reg_loss", 0),
                            self.optimizer.param_groups[0]["lr"],
                        )
                    )
                else:
                    pbar.set_description(
                        "loss={:.4f} , Reg loss{:.4f}), lr={:.6f}".format(
                            loss_val,
                            loss_dict.get("reg_loss", 0),
                            self.optimizer.param_groups[0]["lr"],
                        )
                    )
                pbar.update(loader.batch_size)
            loss_list.append(loss_val)
        self.logger.debug(f"==> Average Loss : {np.mean(loss_list)}")

        # if self.opt.debug:
        #     break

        #! Added by PJ , Its a workaround
        # TODO : Find a permanent solution
        # Normalize the gradient by the number of steps
        if self.opt.accumulate:
            for n_p, p in self.model.named_parameters():
                if p.requires_grad:
                    p.grad = temp_grads[n_p] / self.local_step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # self.optimizer.step()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

        if self.ema is not None:
            self.ema.update()

        average_loss = (total_reg_loss + total_sds_loss) / self.local_step
        self.stats["loss"].append(average_loss)

        wandb.log(
            {
                "epoch": self.epoch,
                "train_one_epoch/loss": average_loss,
                "train_one_epoch/rgb_sds": total_sds_loss / self.local_step,
                "train_one_epoch/reg_loss": total_reg_loss / self.local_step,
                "learning_rate/lr": self.optimizer.param_groups[0]["lr"],
            }
        )

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.logger.info(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.logger.info(f"==> Finished Epoch {self.epoch}.")
        # log with debug level the norm of weights and biases of the pose mlp
        norm_dict = {
            n: p.norm().item()
            for n, p in self.model.named_parameters()
            if "pose_mlp" in n
        }
        norm_dict["epoch"] = self.epoch
        self.logger.debug(f"==> Pose MLP norm: {norm_dict}")
        wandb.log(norm_dict)

    def evaluate_one_epoch(self, loader, name=None):
        self.logger.info(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f"ep{self.epoch:04d}"

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        vis_frames = []
        with torch.no_grad():
            self.local_step = 0

            for i, data in enumerate(loader):
                self.local_step += 1
                if i != 25:
                    continue

                # with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [
                        torch.zeros_like(preds).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    if not self.model.opt.video:
                        pred = (preds[0].detach().cpu().numpy() * 255).astype(np.uint8)
                        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
                        vis_frames.append(pred)
                        pbar.set_description(
                            f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})"
                        )
                        pbar.update(loader.batch_size)
                    else:
                        pred = preds.detach().cpu().numpy()
                        pbar.set_description(
                            f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})"
                        )
                        pbar.update(loader.batch_size)
                if self.write_train_video:
                    if not self.model.opt.video:
                        pred_rgb_only = (preds[:, 256:, :].cpu().numpy() * 255).astype(
                            np.uint8
                        )
                        self.train_video_frames.append(pred_rgb_only)
                    else:
                        pred_np = (pred * 255).astype(np.uint8)
                        t_pred = pred_np.transpose(1, 0, 2, 3)
                        t_pred = t_pred.reshape(t_pred.shape[0], -1, t_pred.shape[3])
                        self.train_video_frames.append(t_pred)

        if not self.model.opt.video and self.epoch % 10 == 0:
            save_path = os.path.join(self.workspace, "validation", f"{name}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, np.hstack(vis_frames))
        else:
            if False:
                save_path = os.path.join(self.workspace, "validation", f"{name}.mp4")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                pred = (pred * 255).astype(np.uint8)
                imageio.mimwrite(save_path, pred, fps=1, quality=5, macro_block_size=1)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == "min" else -result
                )  # if max mode, use -result
            else:
                self.stats["results"].append(
                    average_loss
                )  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.logger.info(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.logger.info(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stats": self.stats,
        }

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:

            state["model"] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(
                    self.ckpt_path, self.stats["checkpoints"].pop(0)
                )
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.logger.info(f" New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.logger.warning(
                    f" no evaluated results found, skip saving best checkpoint."
                )

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.logger.info(f" Latest checkpoint is {checkpoint}")
            else:
                self.logger.info(" No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.logger.info(" loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.logger.info(" loaded model.")
        if len(missing_keys) > 0:
            self.logger.info(f" missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.logger.info(f" unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict["ema"])
                self.logger.info(" loaded EMA.")
            except:
                self.logger.info(" failed to loaded EMA.")

        if model_only:
            return

        self.stats = checkpoint_dict["stats"]
        self.epoch = checkpoint_dict["epoch"]
        self.global_step = checkpoint_dict["global_step"]
        self.logger.info(f" load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.logger.info(" loaded optimizer.")
            except:
                self.logger.warning(" Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.logger.info(" loaded scheduler.")
            except:
                self.logger.warning(" Failed to load scheduler.")

        if self.scaler and "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.logger.info(" loaded scaler.")
            except:
                self.logger.warning(" Failed to load scaler.")
