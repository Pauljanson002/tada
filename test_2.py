import torch
import torchvision
import cv2
from torch import nn
import numpy as np
ckpt = torch.load(
    "/home/paulj/Downloads/best/17/checkpoints/dg_native_resol_17_ep0100.pth"
)
from omegaconf import OmegaConf
from nvdiffrast import torch as dr
config = OmegaConf.load("/home/paulj/Downloads/best/17/config.yaml")
from lib.dlmesh import DLMesh
from torch.nn import functional as F    
model = DLMesh(config.model)
model.load_state_dict(ckpt["model"])
from lib.provider import ViewDataset
model.num_frames = 4
import imageio
dataset = ViewDataset(config.data, "cuda", "test", 4)
for view in range(4):
    data = dataset[view]
    model.to("cuda")
    H, W = 256,256
    h,w = 256,256
    mvp = data["mvp"]
    rays_o = data["rays_o"]  # [B, N, 3]
    rays_d = data["rays_d"]  # [B, N, 3]
    is_train = False


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
                            torch.transpose(mvp.unsqueeze(0), 1, 2)).float() 
    directional  = rays_d.view(1, h, w, 3)

    bg_rast,_ = dr.rasterize(model.glctx, pos_clip, tri, (h, w))
    bg_interp,_ = dr.interpolate(col, bg_rast, tri)
    texture_interp,_ = dr.interpolate(texture_coordinates, bg_rast, tri)
    bg_out = dr.texture(cube_map_texture, directional, boundary_mode="cube")
    bg_color = bg_out

    rgb_frame_list = []
    for i in range(8):
        frame_id = i
        pr_mesh, smplx_landmarks, prediction, smplx_joints = model.get_mesh(
            is_train=is_train, frame_id=frame_id
        )

        albedo_image = cv2.imread("data/mesh_albedo.png")
        albedo_image = cv2.cvtColor(albedo_image, cv2.COLOR_BGR2RGB)
        albedo_image = albedo_image.astype(np.float32) / 255.0
        raw_albedo = torch.as_tensor(albedo_image, dtype=torch.float32, device="cuda")

        pr_mesh.set_albedo(raw_albedo)
        rgb, normal, alpha = model.renderer(
            pr_mesh,
            mvp.unsqueeze(0),
            256,
            256,
            None,
            1.0,
            "albedo",
            model.opt.ssaa,
            is_train=is_train,
        )
        rgb = rgb * alpha + (1 - alpha) * bg_color
        rgb_frame_list.append(rgb)
    rgbt = torch.stack(rgb_frame_list, dim=0)

    # torchvision.utils.save_image(rgb.permute(0,3,1,2), "test_2.png")
    rgbt = (rgbt.detach().cpu().numpy() * 255).astype(np.uint8)
    rgbt = rgbt.squeeze(1)
    imageio.mimwrite(
        f"result_{view}.mp4", rgbt, fps=3, quality=5, macro_block_size=1
    )
