from lib.common.renderer import Renderer
from smplx.body_models import SMPLXLayer
import torch
from lib.provider import ViewDataset
from lib.common.obj import Mesh
import torchvision
import random
import pickle
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from lib.common import utils
from lib.common.obj import compute_normal
import numpy as np

from lib.guidance.sd import StableDiffusion
import math
import lib.rotation_conversions as rc

def normalize_vert(vertices, return_cs=False):
    if isinstance(vertices, np.ndarray):
        vmax, vmin = vertices.max(0), vertices.min(0)
        center = (vmax + vmin) * 0.5
        scale = 1.0 / np.max(vmax - vmin)
    else:  # torch.tensor
        vmax, vmin = vertices.max(0)[0], vertices.min(0)[0]
        center = (vmax + vmin) * 0.5
        scale = 1.0 / torch.max(vmax - vmin)
    if return_cs:
        return (vertices - center) * scale, center, scale
    return (vertices - center) * scale


class Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
        try:
            self.glctx = dr.RasterizeCudaContext()
        except:
            self.glctx = dr.RasterizeGLContext()

    def forward(
        self,
        mesh,
        mvp,
        light_d=None,
    ):
        B = mvp.shape[0]
        v_clip = torch.bmm(
            F.pad(mesh.v, pad=(0, 1), mode="constant", value=1.0)
            .unsqueeze(0)
            .expand(B, -1, -1),
            torch.transpose(mvp, 1, 2),
        ).float()  # [B, N, 4]
        # translate the by 1.0
        res = (512, 512)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)

        ################################################################################
        # Interpolate attributes
        ################################################################################

        # Interpolate world space position
        alpha, _ = dr.interpolate(
            torch.ones_like(v_clip[..., :1]), rast, mesh.f
        )  # [B, H, W, 1]
        depth = rast[..., [2]]  # [B, H, W]

        col = torch.ones_like(v_clip)

        color, _ = dr.interpolate(col, rast, mesh.f)
        vn, _ = compute_normal(v_clip[0, :, :3], mesh.f)
        normal, _ = dr.interpolate(vn[None, ...].float(), rast, mesh.f)

        ambient_ratio = 0.5
        # if shading == "normal":
        #     color = (normal + 1) / 2.0
        # elif shading == "albedo":
        #     color = albedo
        # else:  # lambertian
        lambertian = ambient_ratio + (1 - ambient_ratio) * (
            normal @ light_d.view(-1, 1)
        ).float().clamp(min=0)
        color = color[..., :3] * lambertian.repeat(1, 1, 1, 3)

        # normal = (normal + 1) / 2.0
        # normal = dr.antialias(normal, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]

        return alpha, color

    def get_mlp_texture(self, mesh, mlp_texture, rast, rast_db, res=2048):
        # uv = mesh.vt[None, ...] * 2.0 - 1.0
        uv = mesh.vt[None, ...]

        # pad to four component coordinate
        uv4 = torch.cat(
            (uv, torch.zeros_like(uv[..., 0:1]), torch.ones_like(uv[..., 0:1])), dim=-1
        )

        # rasterize
        _rast, _ = dr.rasterize(self.glctx, uv4, mesh.f.int(), (res, res))
        print("_rast ", _rast.shape)
        # Interpolate world space position
        # gb_pos, _ = dr.interpolate(mesh.v[None, ...], _rast, mesh.f.int())

        # Sample out textures from MLP
        tex = mlp_texture.sample(_rast[..., :-1].view(-1, 3)).view(*_rast.shape[:-1], 3)

        texc, texc_db = dr.interpolate(
            mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs="all"
        )
        print(tex.shape)

        albedo = dr.texture(
            tex, texc, uv_da=texc_db, filter_mode="linear-mipmap-linear"
        )  # [B, H, W, 3]
        # albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device))  # remove background

        # print(tex.shape, albedo.shape)
        # exit()
        return albedo

    @staticmethod
    def get_2d_texture(mesh, rast, rast_db):
        texc, texc_db = dr.interpolate(
            mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs="all"
        )

        albedo = dr.texture(
            mesh.albedo.unsqueeze(0),
            texc,
            uv_da=texc_db,
            filter_mode="linear-mipmap-linear",
        )  # [B, H, W, 3]
        albedo = torch.where(
            rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)
        )  # remove background
        return albedo


class CameraConfig:
    h = 512
    w = 512
    H = 512
    W = 512
    near = 0.01
    far = 1000
    fovy_range = [50, 70]
    radius_range = [1.0, 1.2]
    default_fovy = 60
    default_polar = 90
    default_radius = 1.1
    default_azimuth = 0
    phi_range = [0, 0]
    theta_range = [60, 60]
    head_phi_range = [-30, 30]
    head_theta_range = [75, 85]
    angle_overhead = 30
    angle_front = 90
    dir_text = True
    jitter_pose = False
    uniform_sphere_rate = 0.0
    side_view_only = False


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


if __name__ == "__main__":
    
    with open("4d/poses/running_mean.pkl","rb") as f:
        running_pose = pickle.load(f)
        running_pose = torch.tensor(running_pose).cuda().float()
        running_pose = rc.axis_angle_to_matrix(running_pose.view(-1,3)).view(1, 55, 3, 3)
    
    renderer = Renderer()
    smplx_layer = SMPLXLayer(model_path="./data/smplx/SMPLX_NEUTRAL_2020.npz").cuda()
    smplx_params = running_pose[:,1:22,...]
    # smplx_params[15]["lr"] = 0.01
    dataset = ViewDataset(CameraConfig, "cuda", type="test", size=4)
    faces_tensor = torch.from_numpy(smplx_layer.faces.astype("int32")).int().cuda()
    data = dataset[1]

    guidance = StableDiffusion("cuda", False, False)

    prompt = ["A a man is punching facing forward, full-body"]

    text_embeds = guidance.get_text_embeds(prompt)
    empty_embeds = guidance.get_text_embeds(
        ["low motion, static statue, not moving, no motion"]
    )
    light_d = data["rays_o"][0] + torch.randn(
        3, device=data["rays_o"].device, dtype=torch.float
    )
    light_d = safe_normalize(light_d)

    smplx_output = smplx_layer(body_pose=smplx_params)
    # color

    vertices = normalize_vert(smplx_output.vertices[0])
    mesh = Mesh(vertices, faces_tensor)

    alpha, color = renderer(mesh, data["mvp"][None, ...], light_d=light_d)

    # Rendering ended

    color = color[:, :, :, :3].permute(0, 3, 1, 2)
    torchvision.utils.save_image(color, "color.png")
    image_latent = guidance.encode_imgs(color)
    image_latent.requires_grad = True
    optimizer = torch.optim.AdamW([image_latent], lr=0.1)
    for i in range(1000):
        dir_text_z = [
            empty_embeds[0],
            text_embeds[0],
        ]
        dir_text_z = torch.stack(dir_text_z)
        optimizer.zero_grad()
        loss = guidance.train_step(dir_text_z, image_latent, 100, True)
        print(loss.item())

        loss.backward()
        optimizer.step()
        rgb = guidance.decode_latents(image_latent)
        torchvision.utils.save_image(rgb, "rgb.png")

    print("Test passed")
