import os
import random

import numpy as np
import trimesh
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
# from apps.mp import draw_landmarks_on_image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from lib.encoding import get_encoder
from lib.common.obj import Mesh, safe_normalize, normalize_vert, save_obj_mesh, compute_normal
from lib.common.utils import trunc_rev_sigmoid, SMPLXSeg
from lib.common.renderer import Renderer
from lib.common.remesh import smplx_remesh_mask, subdivide, subdivide_inorder
from lib.common.lbs import warp_points
from lib.common.visual import draw_landmarks
from lib.rotation_conversions import rotation_6d_to_matrix,matrix_to_axis_angle,axis_angle_to_matrix,matrix_to_rotation_6d
import nvdiffrast.torch as dr
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

import torchvision
import pickle
import math
import logging
logger = logging.getLogger(__name__)


def sinusoidal_embedding(dim,frame_limit):
    # pe = torch.FloatTensor(
    #     [
    #         [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
    #         for p in range(frame_limit)
    #     ]
    # )
    pe = torch.FloatTensor(
        [
            [1 * 2**(i//2) * math.pi * p for i in range(dim)] for p in range(frame_limit)
        ]
    )
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        
        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
        
        # init weights with kaiming normalization
        for l in range(num_layers):
            if l != num_layers - 1:
                nn.init.kaiming_normal_(self.net[l].weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.zeros_(self.net[l].weight)
            if bias:
                nn.init.zeros_(self.net[l].bias)


    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

class PoseField(nn.Module):
    def __init__(self, input_dim, hidden_dims,output_dim,frames,pose_mlp_args,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            if i % 2 == 1:
                layers.append(nn.LayerNorm(hidden_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # initialize output layer to output zeros
        if pose_mlp_args.init_zero:
            layers[-1].weight.data.zero_()
            layers[-1].bias.data.zero_()

        self.layers = nn.Sequential(*layers)
        self.create_embedding_fn()
        self.pose_mlp_args = pose_mlp_args
        self.twice_std_dev = torch.load("4d/poses/running_std_dev.pt").cuda().view(126)
        self.max_val = torch.load("4d/poses/running_max.pt").float().cuda().view(126)
        self.min_val = torch.load("4d/poses/running_min.pt").float().cuda().view(126)

    def create_embedding_fn(self):
        embed_fns = []
        d = 1
        out_dim = 0
        if False:
            embed_fns.append(lambda x : x)
            out_dim += d
        max_freq = 5
        N_freqs = 4

        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        embeds = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        # logger.debug(embeds)
        return embeds
    def forward(self, tau,prev_pose=None):
        x = self.embed(tau)
        if prev_pose is not None:
            x = torch.cat([x,prev_pose],dim=-1)
        output = self.layers(x)
        if self.pose_mlp_args.use_clamp == "tanh":
            output = torch.tanh(output/ self.pose_mlp_args.tanh_scale ) * self.pose_mlp_args.tanh_scale
        elif self.pose_mlp_args.use_clamp == "std":
            output = output * self.twice_std_dev/2
        elif self.pose_mlp_args.use_clamp == "maxmin":
            output = torch.clamp(output,self.min_val,self.max_val)
        if self.pose_mlp_args.tau_scale > 0:
            output = output * tau**self.pose_mlp_args.tau_scale
        return output


class AngleField(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, output_dim, pose_mlp_args, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if i % 2 == 1:
                layers.append(nn.LayerNorm(hidden_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        if pose_mlp_args.init_zero:
            layers[-1].weight.data.zero_()
            layers[-1].bias.data.zero_()

        self.layers = nn.Sequential(*layers)
        self.create_embedding_fn()
        self.pose_mlp_args = pose_mlp_args

    def create_embedding_fn(self):
        embed_fns = []
        d = 1
        out_dim = 0
        if False:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = 5
        N_freqs = 4

        embed_type = "nerf"
        if embed_type == "transformer":
            freq_bands = torch.tensor([1 / (10000 ** (2 * i  / 4)) for i in range(4)])
        else:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        embeds = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        return embeds

    def forward(self, joint_id, tau):
        # Embed the inputs

        # joint_embed = self.embed(joint_id)
        # tau_embed = self.embed(tau)
        # both_embed = torch.cat([joint_embed, tau_embed[:,:4]], dim=-1)
        # inputs = both_embed
        joint_tau = torch.cat([joint_id, tau], dim=-1)
        inputs = self.embed(joint_tau)
        output = self.layers(inputs)

        # Scale the output by tau^(0.35) if specified in pose_mlp_args

        # Apply tanh activation and scale if specified in pose_mlp_args
        if self.pose_mlp_args.use_clamp == "tanh":
            output = (
                torch.tanh(output / self.pose_mlp_args.tanh_scale)
                * self.pose_mlp_args.tanh_scale
            )

        if self.pose_mlp_args.tau_scale > 0:
            output = output * tau**self.pose_mlp_args.tau_scale

        return output


class Displacement(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, output_dim, pose_mlp_args, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if i % 2 == 1:
                layers.append(nn.LayerNorm(hidden_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        if pose_mlp_args.init_zero:
            layers[-1].weight.data.zero_()
            layers[-1].bias.data.zero_()

        self.layers = nn.Sequential(*layers)
        self.create_embedding_fn()
        self.pose_mlp_args = pose_mlp_args

    def create_embedding_fn(self):
        embed_fns = []
        d = 1
        out_dim = 0
        if False:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = 5
        N_freqs = 4

        embed_type = "nerf"
        if embed_type == "transformer":
            freq_bands = torch.tensor([1 / (10000 ** (2 * i / 4)) for i in range(4)])
        else:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        embeds = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        return embeds

    def forward(self, position, tau):
        # Embed the inputs

        # joint_embed = self.embed(joint_id)
        # tau_embed = self.embed(tau)
        # both_embed = torch.cat([joint_embed, tau_embed[:,:4]], dim=-1)
        # inputs = both_embed
        joint_tau = torch.cat([position, tau], dim=-1)
        inputs = self.embed(joint_tau)
        output = self.layers(inputs)

        # Scale the output by tau^(0.35) if specified in pose_mlp_args

        # Apply tanh activation and scale if specified in pose_mlp_args
        if self.pose_mlp_args.use_clamp == "tanh":
            output = (
                torch.tanh(output / self.pose_mlp_args.tanh_scale)
                * self.pose_mlp_args.tanh_scale
            )

        if self.pose_mlp_args.tau_scale > 0:
            output = output * tau**self.pose_mlp_args.tau_scale
        output = output + position

        return output


class DLMesh(nn.Module):
    def __init__(self, opt, num_layers_bg=2, hidden_dim_bg=16):

        super(DLMesh, self).__init__()

        self.opt = opt
        self.num_remeshing = 1
        self.vpose = self.opt.vpose
        self.renderer = Renderer()
        self.glctx = dr.RasterizeCudaContext()
        self.device = torch.device("cuda")
        self.lock_beta = opt.lock_beta
        self.intermediate = None
        if self.opt.lock_geo:  # texture
            self.mesh = Mesh.load_obj(self.opt.mesh)
            self.mesh.auto_normal()
        else:  # geometry
            self.body_model = smplx.create(
                model_path="./data/smplx/SMPLX_NEUTRAL_2020.npz",
                model_type='smplx',
                create_global_orient=True,
                create_body_pose=True,
                create_betas=True,
                create_left_hand_pose=True,
                create_right_hand_pose=True,
                create_jaw_pose=True,
                create_leye_pose=True,
                create_reye_pose=True,
                create_expression=True,
                create_transl=False,
                use_pca=False,
                use_face_contour=True,
                flat_hand_mean=True,
                num_betas=300,
                num_expression_coeffs=100,
                num_pca_comps=12,
                dtype=torch.float32,
                batch_size=1,
            ).to(self.device)

            self.smplx_faces = self.body_model.faces.astype(np.int32)

            if self.vpose:
                vp , ps = load_model('V02_05', model_code=VPoser, remove_words_in_model_weights='vp_model.',disable_grad=True)
                self.body_prior = vp.to(self.device)
            for p in self.body_model.parameters():
                p.requires_grad = False
            param_file = "./data/init_body/fit_smplx_params.npz"
            smplx_params = dict(np.load(param_file))
            self.betas = torch.as_tensor(smplx_params["betas"]).to(self.device)
            self.jaw_pose = torch.as_tensor(smplx_params["jaw_pose"]).to(self.device)
            self.num_frames = opt.num_frames
            self.body_pose = torch.as_tensor(smplx_params["body_pose"]).to(self.device)

            self.body_pose = self.body_pose.view(-1, 3)
            self.body_pose[[0, 1, 3, 4, 6, 7], :2] *= 0
            self.body_pose = self.body_pose.view(1, -1)
            self.add_fake_movement = self.opt.add_fake_movement
            if self.opt.initialize_pose == "diving":
                self.diving_body_pose = pickle.load(open("4d/poses/diving.pkl", "rb"))["body_pose"]
            elif self.opt.initialize_pose == "running":
                self.diving_body_pose = pickle.load(open("4d/poses/running.pkl", "rb"))["body_pose"]
            elif self.opt.initialize_pose == "running_gauss":
                self.diving_body_pose = pickle.load(open("4d/poses/running.pkl", "rb"))["body_pose"]
                self.diving_body_pose += np.random.normal(0, 0.1, self.diving_body_pose.shape)
            elif self.opt.initialize_pose == "running_first_frame":
                self.diving_body_pose = pickle.load(open("4d/poses/running.pkl", "rb"))["body_pose"]
                self.diving_body_pose = self.diving_body_pose[:1, :]
                self.diving_body_pose = np.repeat(self.diving_body_pose, self.num_frames, axis=0)
            elif self.opt.initialize_pose == "zero":
                self.diving_body_pose = np.zeros((self.num_frames, 63))
            elif self.opt.initialize_pose == "running_mean":
                self.diving_body_pose = pickle.load(open("4d/poses/running_mean.pkl","rb"))
                self.diving_body_pose = np.repeat(
                    self.diving_body_pose, self.num_frames, axis=0
                )

            self.diving_body_pose = torch.as_tensor(self.diving_body_pose).float().to(self.device)
            self.diving_body_pose = self.diving_body_pose[:self.num_frames,:]
            if self.opt.use_6d:
                if self.opt.model_change:
                    if self.opt.use_full_pose:
                        self.init_full_pose_6d = torch.cat([
                            torch.zeros(1, 6).to(self.device), # global_orient
                            matrix_to_rotation_6d(axis_angle_to_matrix(torch.as_tensor(smplx_params["body_pose"]).view(-1, 3)).to(self.device)), # body
                            matrix_to_rotation_6d(axis_angle_to_matrix(torch.as_tensor(smplx_params["jaw_pose"]).view(-1, 3)).to(self.device)), # jaw
                            torch.zeros(1, 6).to(self.device),  # left eye
                            torch.zeros(1, 6).to(self.device), # right eye
                            torch.zeros(15, 6).to(self.device), # left hand
                            torch.zeros(15, 6).to(self.device), # right hand
                        ],dim=0).reshape(1,-1)
                        self.full_pose_6d = torch.zeros(self.init_full_pose_6d.shape).to(self.device) 
                    else:
                        if self.vpose:
                            if not self.opt.video:
                                self.body_pose_6d = torch.zeros([1,32]).to(self.device)
                            else:
                                self.init_body_pose_6d_set = self.body_prior.encode(self.diving_body_pose).mean # latent space
                                # self.init_body_pose_6d_set = torch.randn(self.diving_body_pose.shape[0],32).to(self.device)
                                self.body_pose_6d_set = torch.zeros(self.init_body_pose_6d_set.shape).to(self.device)

                        else:
                            self.init_body_pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(self.body_pose.view(-1, 21, 3))).view(1, -1)
                            self.init_body_pose_6d_set = self.init_body_pose_6d.repeat([self.num_frames,1])
                            self.init_body_pose_6d_set = matrix_to_rotation_6d(axis_angle_to_matrix(self.diving_body_pose.view(-1,3))).view(self.num_frames,-1).float()
                            self.prev_pose = self.init_body_pose_6d_set
                            self.prev_pose = torch.cat([self.prev_pose,self.prev_pose[None,0]],dim=0)
                            self.body_pose_6d = torch.zeros(self.init_body_pose_6d.shape).to(self.device)
                            self.body_pose_6d_set = torch.zeros(self.init_body_pose_6d_set.shape).to(self.device)
                else:
                    if self.opt.use_full_pose:
                        self.full_pose_6d = torch.cat([
                            torch.zeros(1, 6).to(self.device), # global_orient
                            matrix_to_rotation_6d(axis_angle_to_matrix(torch.as_tensor(smplx_params["body_pose"]).view(-1, 3)).to(self.device)), # body
                            matrix_to_rotation_6d(axis_angle_to_matrix(torch.as_tensor(smplx_params["jaw_pose"]).view(-1, 3)).to(self.device)), # jaw
                            torch.zeros(1, 6).to(self.device),  # left eye
                            torch.zeros(1, 6).to(self.device), # right eye
                            torch.zeros(15, 6).to(self.device), # left hand
                            torch.zeros(15, 6).to(self.device), # right hand
                        ],dim=0).reshape(1,-1)
                    else:
                        if self.vpose:
                            # self.init_body_pose_6d_set = torch.randn(self.diving_body_pose.shape[0],32).to(self.device)
                            if self.opt.initialize_pose == "zero":
                                if not self.opt.video:
                                    self.body_pose_6d = torch.zeros([1,32]).to(self.device)
                                else:
                                    self.body_pose_6d_set = torch.zeros([self.num_frames,32]).to(self.device)
                            else:
                                self.body_pose_6d_set = self.body_prior.encode(self.diving_body_pose).mean # latent space

                        else:
                            self.body_pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(self.body_pose.view(-1, 21, 3))).view(1, -1)
                self.body_pose = None

            self.global_orient = torch.as_tensor(smplx_params["global_orient"]).to(self.device)

            self.expression = torch.zeros(1, 100).to(self.device)

            self.remesh_mask = self.get_remesh_mask()
            self.faces_list, self.dense_lbs_weights, self.uniques, self.vt, self.ft = self.get_init_body()

            N = self.dense_lbs_weights.shape[0]

            self.simplify = self.opt.simplify

            if self.simplify:
                self.opt.pose_mlp = "displacement"

            if self.opt.video:
                if self.vpose:
                    if self.opt.pose_mlp == "pose":
                        self.pose_mlp = PoseField(8, [32,32], 32,self.num_frames,self.opt.pose_mlp_args)
                    elif self.opt.pose_mlp == "angle":
                        raise NotImplementedError("AngleField is not implemented yet for vpose")
                else:
                    if self.opt.pose_mlp == "pose":
                        if self.vpose:
                            self.pose_mlp = PoseField(8, [32,32], 32,self.num_frames,self.opt.pose_mlp_args)
                        else:
                            self.pose_mlp = PoseField(
                                8, [32, 32], 126, self.num_frames, self.opt.pose_mlp_args
                            )
                    elif self.opt.pose_mlp == "kickstart":
                        self.pose_mlp = PoseField(
                            126+8, [128, 128], 126, self.num_frames, self.opt.pose_mlp_args
                        )
                    else:
                        self.pose_mlp = AngleField(
                            32, [64, 64,64,64], 6, self.opt.pose_mlp_args
                        )

            else:
                if self.vpose:
                    if self.opt.pose_mlp == "pose":
                        self.pose_mlp = PoseField(8, [32,32], 32)
                    elif self.opt.pose_mlp == "angle":
                        raise NotImplementedError("AngleField is not implemented yet for vpose")
                else:
                    if self.opt.pose_mlp == "pose":
                        self.pose_mlp = PoseField(
                            8, [32, 32], 126,self.num_frames,self.opt.pose_mlp_args
                        )
                    elif self.opt.pose_mlp == "angle":
                        self.pose_mlp = AngleField(
                            16, [64, 64, 64, 64], 6, self.opt.pose_mlp_args
                        )
                    else:
                        self.pose_mlp = Displacement(
                            32, [64, 64, 64, 64], 3, self.opt.pose_mlp_args
                        )
            self.pose_mlp = self.pose_mlp.to(self.device)

        # background network
        if not self.opt.skip_bg:
            self.encoder_bg, in_dim_bg = get_encoder('frequency_torch', multires=4)
            self.bg_net = MLP(in_dim_bg, 3, hidden_dim_bg, num_layers_bg)

        self.mlp_texture = None
        if not self.opt.lock_tex:  # texture parameters
            if self.opt.tex_mlp:
                # self.encoder_tex, self.in_dim = get_encoder('hashgrid', interpolation='smoothstep')
                # self.tex_net = MLP(self.in_dim, 3, 32, 2)
                from .mlptexture import MLPTexture3D
                self.mlp_texture = MLPTexture3D()
            else:
                if False:
                    res = self.opt.albedo_res
                    albedo = torch.ones((res, res, 3), dtype=torch.float32) * 0.5  # default color
                    self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(albedo))
                else:
                    albedo_image = cv2.imread("data/mesh_albedo.png")
                    albedo_image = cv2.cvtColor(albedo_image, cv2.COLOR_BGR2RGB)
                    albedo_image = albedo_image.astype(np.float32) / 255.0
                    self.raw_albedo = torch.as_tensor(albedo_image, dtype=torch.float32, device=self.device)
                    self.raw_albedo = nn.Parameter(self.raw_albedo)
        else:
            albedo_image = cv2.imread("data/mesh_albedo.png")
            albedo_image = cv2.cvtColor(albedo_image, cv2.COLOR_BGR2RGB)
            albedo_image = albedo_image.astype(np.float32) / 255.0
            self.raw_albedo = torch.as_tensor(albedo_image, dtype=torch.float32, device=self.device)

        # Geometry parameters
        if not self.opt.lock_geo:
            # displacement
            if self.opt.geo_mlp:
                self.encoder_geo, in_dim_geo = get_encoder('hashgrid', interpolation='smoothstep')
                self.geo_net = MLP(in_dim_geo, 1, 32, 2)
            else:
                if self.opt.pose_mlp is not None:
                    self.v_offsets =torch.zeros(N, 1,requires_grad=False).to(self.device)
                else:
                    self.v_offsets = nn.Parameter(torch.zeros(N, 3))
            # shape
            if not self.lock_beta:
                self.betas = nn.Parameter(self.betas)
            # expression
            rich_data = np.load("./data/talkshow/rich.npy")
            self.rich_params = torch.as_tensor(rich_data, dtype=torch.float32, device=self.device)
            if not self.opt.lock_expression:
                self.expression = nn.Parameter(self.expression)
            # self.jaw_pose = nn.Parameter(self.jaw_pose)
        if not self.opt.lock_pose:
            if self.opt.pose_mlp == "none":
                if self.opt.use_6d:
                    if self.opt.use_full_pose:
                        self.full_pose_6d = nn.Parameter(self.full_pose_6d)
                    else:
                        if not self.opt.video:
                            self.body_pose_6d = nn.Parameter(self.body_pose_6d)
                        else:
                            self.body_pose_6d_set = nn.Parameter(self.body_pose_6d_set)
                else:
                    self.body_pose = nn.Parameter(self.body_pose)

        # Create an FaceLandmarker object.
        base_options = python.BaseOptions(model_asset_path='data/mediapipe/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.joint_locations = torch.load("a.pt").to(self.device).squeeze(0)
        self.count = 0
        self.shading = self.opt.shading

    @torch.no_grad()
    def get_init_body(self, cache_path='./data/init_body/data.npz'):
        if True:
            if self.num_remeshing == 1:
                cache_path = './data/init_body/data.npz'
                data = np.load(cache_path)
                faces_list = [torch.as_tensor(data['dense_faces'], device=self.device)]
                dense_lbs_weights = torch.as_tensor(data['dense_lbs_weights'], device=self.device)
                unique_list = [data['unique']]
                vt = torch.as_tensor(data['vt'], device=self.device)
                ft = torch.as_tensor(data['ft'], device=self.device)
            else:
                cache_path = './data/init_body/data-remesh2.npz'
                data = np.load(cache_path, allow_pickle=True)
                faces_list = [torch.as_tensor(f, device=self.device) for f in data["faces"]]
                dense_lbs_weights = torch.as_tensor(data['dense_lbs_weights'], device=self.device)
                unique_list = data['uniques']
                vt = torch.as_tensor(data['vt'], device=self.device)
                ft = torch.as_tensor(data['ft'], device=self.device)
        else:
            output = self.body_model(
                betas=self.betas,
                body_pose=self.body_pose,
                jaw_pose=self.jaw_pose,
                expression=self.expression,
                return_verts=True
            )
            v_cano = output.v_posed[0]

            # re-meshing
            dense_v_cano, dense_faces, dense_lbs_weights, unique = subdivide(v_cano.cpu().numpy(),
                                                                             self.smplx_faces[self.remesh_mask],
                                                                             self.body_model.lbs_weights.detach().cpu().numpy())
            dense_faces = np.concatenate([dense_faces, self.smplx_faces[~self.remesh_mask]])

            unique_list = [unique]
            faces_list = [dense_faces]
            # re-meshing
            for _ in range(1, self.num_remeshing):
                dense_v_cano, dense_faces, dense_lbs_weights, unique = subdivide(dense_v_cano, dense_faces,
                                                                                 dense_lbs_weights)
                unique_list.append(unique)
                faces_list.append(dense_faces)

            dense_v = torch.as_tensor(dense_v_cano, device=self.device)
            dense_faces = torch.as_tensor(dense_faces, device=self.device)
            dense_lbs_weights = torch.as_tensor(dense_lbs_weights, device=self.device)

            dense_v_posed = warp_points(dense_v, dense_lbs_weights, output.joints_transform[:, :55])[0]

            dense_v_posed = normalize_vert(dense_v_posed)

            vt, ft = Mesh(device=self.device).auto_uv(v=dense_v_posed, f=dense_faces)

            np.savez(
                cache_path,
                faces=np.array(faces_list, dtype=object),
                dense_lbs_weights=dense_lbs_weights.cpu().numpy(),
                uniques=np.array(unique_list, dtype=object),
                vt=vt.cpu().numpy(),
                ft=ft.cpu().numpy()
            )

            # trimesh.Trimesh(dense_v_posed.cpu().numpy(), dense_faces.cpu().numpy()).export("mesh.obj")
            # exit()

            # exit()

        return faces_list, dense_lbs_weights, unique_list, vt, ft

    def get_remesh_mask(self):
        ids = list(set(SMPLXSeg.front_face_ids) - set(SMPLXSeg.forehead_ids))
        ids = ids + SMPLXSeg.ears_ids + SMPLXSeg.eyeball_ids + SMPLXSeg.hands_ids
        mask = ~np.isin(np.arange(10475), ids)
        mask = mask[self.body_model.faces].all(axis=1)
        return mask

    def get_params(self, lr):
        params = []

        # #!!!!!!!! temp
        # params.append({"params": self.displacement.parameters(), "lr": lr})

        if not self.opt.skip_bg: # default skip_bg = True
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        if not self.opt.lock_tex: # default lock_tex = False
            if self.opt.tex_mlp: # default tex_mlp = False
                params.extend([
                    {'params': self.mlp_texture.parameters(), 'lr': lr * 10},
                ])
            else:
                params.append({'params': self.raw_albedo, 'lr': lr * 10})

        if not self.opt.lock_geo:
            if self.opt.geo_mlp:
                params.extend([
                    {'params': self.encoder_geo.parameters(), 'lr': lr * 10},
                    {'params': self.geo_net.parameters(), 'lr': lr},
                ])
            else:
                if False:
                    params.append({'params': self.v_offsets, 'lr': 0.0001})

            if not self.lock_beta:
                params.append({'params': self.betas, 'lr': 0.1})

            if not self.opt.lock_expression:
                params.append({'params': self.expression, 'lr': 0.05})

        if not self.opt.lock_pose:

            if self.opt.pose_mlp != "none":
                params.append({'params': self.pose_mlp.parameters(), 'lr': lr})
            elif self.opt.use_6d:
                if self.opt.use_full_pose:
                    params.append({'params': self.full_pose_6d, 'lr': 0.05})
                else:
                    if not self.opt.video:
                        params.append({'params': self.body_pose_6d, 'lr': lr})
                    else:
                        params.append({'params': self.body_pose_6d_set, 'lr': lr})
            else:   
                params.append({'params': self.body_pose, 'lr': 0.05})
            #!!!! Not training Jaw pose for now
            # params.append({'params': self.jaw_pose, 'lr': 0.05})

        return params

    def get_vertex_offset(self, is_train):
        v_offsets = self.v_offsets
        if not is_train and self.opt.replace_hands_eyes:
            v_offsets[SMPLXSeg.eyeball_ids] = 0.
            v_offsets[SMPLXSeg.hands_ids] = 0.
        return v_offsets

    def get_mesh(self, is_train,frame_id=0):
        # os.makedirs("./results/pipline/obj/", exist_ok=True)
        video = self.opt.video
        global_orient = self.global_orient
        jaw_pose = None
        left_eye_pose = None
        right_eye_pose = None
        left_hand_pose = None
        right_hand_pose = None
        if not self.opt.lock_geo:
            if self.opt.pose_mlp != "none" and self.opt.pose_mlp != "displacement": 
                if not video: # image case
                    # TODO: Implement the pose mlp for non vpose case for image
                    if self.opt.model_change:
                        pose_mlp_output = self.pose_mlp(
                            torch.tensor([0], device=self.device)
                        )
                        prediction = (
                            pose_mlp_output + self.init_body_pose_6d
                        )
                        body_pose = matrix_to_axis_angle(
                            rotation_6d_to_matrix(prediction.view(-1, 21, 6))
                        ).view(1, -1)
                    else:
                        prediction = self.pose_mlp(self.body_pose_6d)
                        body_pose = self.body_prior.decode(prediction.unsqueeze(0))['pose_body'].contiguous().view(1,-1)
                else:
                    # Video case , with pose mlp
                    if self.opt.pose_mlp == "pose":
                        pose_mlp_output = self.pose_mlp(torch.tensor([frame_id],device=self.device))
                    elif self.opt.pose_mlp == "kickstart":
                        pose_mlp_output = self.pose_mlp(torch.tensor([frame_id],device=self.device),self.prev_pose[frame_id])
                        self.prev_pose[frame_id+1] = pose_mlp_output.detach()
                    elif self.opt.pose_mlp == "angle":
                        # angle_batch = torch.arange(0,21,device=self.device).unsqueeze(1)
                        joint_batch = self.joint_locations
                        frame_batch = torch.tensor([frame_id],device=self.device).repeat(21,1)
                        pose_mlp_output = self.pose_mlp(joint_batch,frame_batch)
                        pose_mlp_output = pose_mlp_output.view(1,-1)
                    if self.opt.model_change:
                        prediction = torch.clamp(pose_mlp_output + self.init_body_pose_6d_set[frame_id],self.pose_mlp.min_val,self.pose_mlp.max_val)
                    else:
                        prediction = pose_mlp_output
                    if self.vpose:
                        body_pose = self.body_prior.decode(prediction.unsqueeze(0))['pose_body'].contiguous().view(1,-1)
                    else:
                        body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(prediction.view(-1,21,6))).view(1,-1)
            # Non pose mlp case
            elif self.opt.use_6d:
                if self.opt.model_change:
                    if self.opt.use_full_pose:
                        full_pose_6d = self.full_pose_6d + self.init_full_pose_6d
                    else:
                        if not video:
                            body_pose_6d = self.body_pose_6d + self.init_body_pose_6d
                        else:
                            # body_pose_6d_set = self.body_pose_6d_set + self.init_body_pose_6d_set
                            body_pose_6d = self.body_pose_6d_set[frame_id] + self.init_body_pose_6d_set[frame_id]
                else:
                    if self.opt.use_full_pose:
                        full_pose_6d = self.full_pose_6d
                    else:
                        if not video:
                            body_pose_6d = self.body_pose_6d
                        else:
                            body_pose_6d = self.body_pose_6d_set[frame_id]
                if self.opt.use_full_pose:
                    full_pose_6d = full_pose_6d.view(-1,6)
                    global_orient = matrix_to_axis_angle(rotation_6d_to_matrix(full_pose_6d[:1].view(-1,6))).view(1,-1)
                    body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(full_pose_6d[1:22].view(-1,6))).view(1,-1)
                    jaw_pose = matrix_to_axis_angle(rotation_6d_to_matrix(full_pose_6d[22:23].view(-1,6))).view(1,-1)
                    left_eye_pose = matrix_to_axis_angle(rotation_6d_to_matrix(full_pose_6d[23:24].view(-1,6))).view(1,-1)
                    right_eye_pose = matrix_to_axis_angle(rotation_6d_to_matrix(full_pose_6d[24:25].view(-1,6))).view(1,-1)
                    left_hand_pose = matrix_to_axis_angle(rotation_6d_to_matrix(full_pose_6d[25:40].view(-1,6))).view(1,-1)
                    right_hand_pose = matrix_to_axis_angle(rotation_6d_to_matrix(full_pose_6d[40:55].view(-1,6))).view(1,-1)
                else:
                    if self.vpose:
                        body_pose = self.body_prior.decode(body_pose_6d.unsqueeze(0))['pose_body'].contiguous().view(1,-1)
                    else:
                        body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_6d.view(-1,21,6))).view(1,-1)
                    global_orient = self.global_orient
                    jaw_pose = None
                    left_eye_pose = None
                    right_eye_pose = None
                    left_hand_pose = None
                    right_hand_pose = None
                prediction = body_pose
            else:
                body_pose = self.body_pose
                global_orient = self.global_orient
                jaw_pose = None
                left_eye_pose = None
                right_eye_pose = None
                left_hand_pose = None
                right_hand_pose = None
            output = self.body_model(
                betas=self.betas,
                body_pose=body_pose,
                jaw_pose=jaw_pose,
                global_orient=global_orient,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                left_eye_pose=left_eye_pose,
                right_eye_pose=right_eye_pose,
                # jaw_pose=random.choice(self.rich_params)[None, :3],
                # jaw_pose=self.rich_params[500:501, :3],
                expression=self.expression,
                return_verts=True
            )
            v_cano = output.v_posed[0]
            landmarks = output.joints[0, -68:, :]
            joints = output.joints[0, :-68, :]
            # re-mesh
            if not self.simplify:
                v_cano_dense = subdivide_inorder(v_cano, self.smplx_faces[self.remesh_mask], self.uniques[0])

                for unique, faces in zip(self.uniques[1:], self.faces_list[:-1]):
                    v_cano_dense = subdivide_inorder(v_cano_dense, faces, unique)

                # #add offset before warp
                if not self.opt.lock_geo:
                    if self.v_offsets.shape[1] ==1:
                        vn = compute_normal(v_cano_dense, self.faces_list[-1])[0]
                        v_cano_dense += self.get_vertex_offset(is_train) * vn
                    else:
                        v_cano_dense += self.get_vertex_offset(is_train)
                # LBS
                v_posed_dense = warp_points(v_cano_dense, self.dense_lbs_weights,
                                            output.joints_transform[:, :55]).squeeze(0)
                # # if not is_train:
                # v_posed_dense = v_cano
                v_posed_dense, center, scale = normalize_vert(v_posed_dense, return_cs=True)
                mesh = Mesh(v_posed_dense, self.faces_list[-1].int(), vt=self.vt, ft=self.ft)
                # mesh = Mesh(
                #     v_posed_dense,
                #     torch.from_numpy(self.body_model.faces.astype(np.int32)).cuda())
                # vt, ft = mesh.auto_uv()
                # breakpoint()
            else:
                import fast_simplification
                import xatlas
                v_cano, center , scale  =  normalize_vert(
                    v_cano, return_cs=True
                )
                vertices,faces = fast_simplification.simplify(v_cano.detach().cpu().numpy(),self.smplx_faces,0.9)
                mesh_trimesh = trimesh.Trimesh(vertices,faces)
                vmapping,indices,uvs = xatlas.parametrize(mesh_trimesh.vertices,mesh_trimesh.faces)
                uvs = torch.tensor(uvs).cuda()
                indices = torch.tensor(indices.astype(int)).cuda().to(torch.int32)
                vertices = torch.from_numpy(vertices).cuda().float()
                frame_batch = torch.tensor([frame_id+1], device=self.device).repeat(
                    vertices.shape[0], 1
                )
                vertices = self.pose_mlp(vertices,frame_batch.cuda())
                mesh = Mesh(vertices,torch.from_numpy(faces).cuda(),vt=uvs,ft=indices)
            mesh.auto_normal()
            # if not self.opt.lock_tex and not self.opt.tex_mlp:
            mesh.set_albedo(self.raw_albedo)
        else:
            mesh = Mesh(base=self.mesh)
            mesh.set_albedo(self.raw_albedo)
        return mesh, landmarks ,prediction,joints

    @torch.no_grad()
    def get_mesh_center_scale(self, phrase):
        assert phrase in ["face", "body"]
        vertices = self.body_model(
            betas=self.betas,
            body_pose=self.body_pose,
            jaw_pose=self.jaw_pose,
            expression=self.expression,
            return_verts=True).vertices[0]
        vertices = normalize_vert(vertices)

        if phrase == "face":
            vertices = vertices[SMPLXSeg.head_ids + SMPLXSeg.neck_ids]
        max_v = vertices.max(0)[0]
        min_v = vertices.min(0)[0]
        scale = (max_v[1] - min_v[1])
        center = (max_v + min_v) * 0.5
        # center = torch.mean(points, dim=0, keepdim=True)
        return center, scale

    @torch.no_grad()
    def export_mesh(self, save_dir):
        # TODO: Export mesh for video
        mesh = self.get_mesh(is_train=False)[0]
        obj_path = os.path.join(save_dir, 'mesh.obj')
        mesh.write(obj_path)

    @torch.no_grad()
    def get_mediapipe_landmarks(self, image):
        """
        Parameters
        ----------
        image: np.ndarray HxWxC

        Returns
        -------
        face_landmarks_list
        """
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image.astype(np.uint8))
        detection_result = self.detector.detect(image_mp)
        face_landmarks_list = detection_result.face_landmarks
        return face_landmarks_list

    def forward(self, rays_o, rays_d, mvp, h, w, light_d=None, ambient_ratio=1.0, shading='albedo', is_train=True):

        batch = rays_o.shape[0]

        if not self.opt.skip_bg:
            dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            bg_color = torch.sigmoid(self.bg_net(self.encoder_bg(dirs.view(-1, 3)))).view(batch, h, w, 3).contiguous()
        else:
            bg_color = torch.ones(batch, h, w, 3).to(mvp.device)

        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=rays_o.device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        if self.opt.use_cubemap:

            # [-1.0,  1.0, -1.0,],
            # [-1.0, -1.0, -1.0,],
            #  [1.0, -1.0, -1.0],
            #  [1.0, -1.0, -1.0],
            #  [1.0,  1.0, -1.0],
            # [-1.0,  1.0, -1.0],

            # -1.0f, -1.0f,  1.0f,
            # -1.0f, -1.0f, -1.0f,
            # -1.0f,  1.0f, -1.0f,
            # -1.0f,  1.0f, -1.0f,
            # -1.0f,  1.0f,  1.0f,
            # -1.0f, -1.0f,  1.0f,

            #  1.0f, -1.0f, -1.0f,
            #  1.0f, -1.0f,  1.0f,
            #  1.0f,  1.0f,  1.0f,
            #  1.0f,  1.0f,  1.0f,
            #  1.0f,  1.0f, -1.0f,
            #  1.0f, -1.0f, -1.0f,

            # -1.0f, -1.0f,  1.0f,
            # -1.0f,  1.0f,  1.0f,
            #  1.0f,  1.0f,  1.0f,
            #  1.0f,  1.0f,  1.0f,
            #  1.0f, -1.0f,  1.0f,
            # -1.0f, -1.0f,  1.0f,

            # -1.0f,  1.0f, -1.0f,
            #  1.0f,  1.0f, -1.0f,
            #  1.0f,  1.0f,  1.0f,
            #  1.0f,  1.0f,  1.0f,
            # -1.0f,  1.0f,  1.0f,
            # -1.0f,  1.0f, -1.0f,

            # -1.0f, -1.0f, -1.0f,
            # -1.0f, -1.0f,  1.0f,
            #  1.0f, -1.0f, -1.0f,
            #  1.0f, -1.0f, -1.0f,
            # -1.0f, -1.0f,  1.0f,
            #  1.0f, -1.0f,  1.0f
            pos = torch.tensor([[-1., -1., -1.],
            [ 1., -1., -1.],
            [ 1.,  1., -1.],
            [-1.,  1., -1.],
            [-1., -1.,  1.],
            [ 1., -1.,  1.],
            [ 1.,  1.,  1.],
            [-1.,  1.,  1.]],dtype=torch.float32).cuda()

            texture_coordinates = torch.tensor([[-1., -1., -1.],
            [ 1., -1., -1.],
            [ 1.,  1., -1.],
            [-1.,  1., -1.],
            [-1., -1.,  1.],
            [ 1., -1.,  1.],
            [ 1.,  1.,  1.],
            [-1.,  1.,  1.]],dtype=torch.float32).cuda()

            tri = torch.tensor([
                [0, 1, 2], [0, 2, 3],  # Front face
                [4, 5, 6], [4, 6, 7],  # Back face
                [0, 1, 5], [0, 5, 4],  # Left face
                [2, 3, 7], [2, 7, 6],  # Right face
                [0, 3, 7], [0, 7, 4],  # Top face
                [1, 2, 6], [1, 6, 5]   # Bottom face
            ], dtype=torch.int32).cuda()

            col = torch.tensor([[-1., -1., -1.],
            [ 1., -1., -1.],
            [ 1.,  1., -1.],
            [-1.,  1., -1.],
            [-1., -1.,  1.],
            [ 1., -1.,  1.],
            [ 1.,  1.,  1.],
            [-1.,  1.,  1.]],dtype=torch.float32).cuda()

            from PIL import Image
            map_texture_locations = ["posx.jpg", "negx.jpg", "posy.jpg", "negy.jpg", "posz.jpg", "negz.jpg"]
            cube_map_texture = torch.zeros((1, 6, 256, 256, 3), dtype=torch.float32)
            for i in range(6):
                cube_map_texture[0, i] = torch.tensor(np.array(Image.open(f"cubemaps/SanFrancisco4/{map_texture_locations[i]}").resize((256, 256))).astype(np.float32) / 255.0)
            cube_map_texture = cube_map_texture.cuda()

            pos_clip = torch.bmm(F.pad(pos, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(1, -1, -1),
                                    torch.transpose(mvp, 1, 2)).float() 
            directional  = rays_d.view(1, h, w, 3)

            bg_rast,_ = dr.rasterize(self.glctx, pos_clip, tri, (h, w))
            bg_interp,_ = dr.interpolate(col, bg_rast, tri)
            texture_interp,_ = dr.interpolate(texture_coordinates, bg_rast, tri)
            bg_out = dr.texture(cube_map_texture, directional, boundary_mode="cube")
            bg_color = bg_out

        # render
        video = self.opt.video
        if video:
            frame_size = self.num_frames
            rgb_frame_list = []
            normal_frame_list = []
            smplx_joints_frame_list = []
            smplx_3d_joints_frame_list = []
            prediction_list = []
            for i in range(frame_size):
                pr_mesh, smplx_landmarks,prediction,smplx_joints = self.get_mesh(is_train=is_train,frame_id=i)
                smplx_3d_joints_frame_list.append(smplx_joints)
                if self.add_fake_movement:
                    # logger.debug(f"Adding fake movement to frame {i}")
                    pr_mesh.v -= torch.tensor([0.0,0,0.25]).cuda()
                    pr_mesh.v += torch.tensor([0.0,0,0.025 * i]).cuda() 
                rgb,normal,alpha = self.renderer(pr_mesh, mvp, h, w, light_d, ambient_ratio, shading, self.opt.ssaa,
                                            mlp_texture=self.mlp_texture, is_train=is_train)
                rgb = rgb * alpha + (1 - alpha) * bg_color
                normal = normal * alpha + (1 - alpha) * bg_color
                smplx_joints = torch.bmm(F.pad(smplx_joints, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(1,-1,-1),torch.transpose(mvp, 1, 2)).float()  # [B, N, 4]

                smplx_joints = smplx_joints[..., :2] / smplx_joints[..., 3:]
                smplx_joints = smplx_joints * 0.5 + 0.5
                rgb_frame_list.append(rgb)
                normal_frame_list.append(normal)
                smplx_joints_frame_list.append(smplx_joints)
                prediction_list.append(prediction)
            rgbt = torch.stack(rgb_frame_list,dim=0)
            normalt = torch.stack(normal_frame_list,dim=0)
            smplx_joints = torch.stack(smplx_joints_frame_list,dim=0)
            smplx_3d_joints = torch.stack(smplx_3d_joints_frame_list,dim=0)
            # torch.save(smplx_3d_joints,"smplx_3d_joints.pt")
            # torch.save(smplx_joints,f"smplx_joints_{self.count}.pt")
            # self.count+=1
            prediction = torch.stack(prediction_list,dim=0)

        else:
            pr_mesh, smplx_landmarks,prediction,joints = self.get_mesh(is_train=is_train)
            rgb, normal, alpha = self.renderer(pr_mesh, mvp, h, w, light_d, ambient_ratio, shading, self.opt.ssaa,
                                            mlp_texture=self.mlp_texture, is_train=is_train)
            rgb = rgb * alpha + (1 - alpha) * bg_color

            normal = normal * alpha + (1 - alpha) * bg_color

            # smplx landmarks
            smplx_landmarks = torch.bmm(F.pad(smplx_landmarks, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0),
                                        torch.transpose(mvp, 1, 2)).float()  # [B, N, 4]
            smplx_landmarks = smplx_landmarks[..., :2] / smplx_landmarks[..., 2:3]
            smplx_landmarks = smplx_landmarks * 0.5 + 0.5
        if video:
            return {
                "video": rgbt,
                "alpha_vid": alpha,
                "normal_vid": normalt,
                "smplx_joints_vid": smplx_joints,
                "smplx_3d_joints_vid": smplx_3d_joints,
                "prediction": prediction
            }
        else:
            return {
                "image": rgb,
                "alpha": alpha,
                "normal": normal,
                "smplx_landmarks": smplx_landmarks,
                "prediction": prediction
            }#
