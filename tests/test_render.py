from calendar import c
import pickle

from cv2 import norm
import cv2
import pyrender
import trimesh
from lib.common.renderer import Renderer
from smplx.body_models import SMPLXLayer
import torch
from lib.provider import ViewDataset
from lib.common.obj import Mesh
import torchvision
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from lib.common import utils
from lib.common.obj import compute_normal
import numpy as np
import lib.rotation_conversions as rc
from lib.guidance.sd import StableDiffusion
import math
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


def normalize_vert(vertices, return_cs=False):
    if isinstance(vertices, np.ndarray):
        vmax, vmin = vertices.max(0), vertices.min(0)
        center = (vmax + vmin) * 0.5
        # Use the maximum extent across all dimensions
        scale = 1.0 / max(vmax - vmin)
    else:  # torch.tensor
        vmax, vmin = vertices.max(0)[0], vertices.min(0)[0]
        center = (vmax + vmin) * 0.5
        # Use the maximum extent across all dimensions
        scale = 1.0 / (vmax - vmin).max()

    normalized = (vertices - center) * scale
    if return_cs:
        return normalized, center, scale
    return normalized


class Renderer2(torch.nn.Module):
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
        texture=None,
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
        res = (256,256)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)

        ################################################################################
        # Interpolate attributes
        ################################################################################

        # Interpolate world space position
        alpha, _ = dr.interpolate(
            torch.ones_like(v_clip[..., :1]), rast, mesh.f
        )  # [B, H, W, 1]
        depth = rast[..., [2]]  # [B, H, W]
        if False:
            col = texture
            
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
            color = color[...,:3] * lambertian.repeat(1, 1, 1, 3)
        else:
            albedo = self.get_2d_texture(mesh, rast, rast_db)

        # normal = (normal + 1) / 2.0
        # normal = dr.antialias(normal, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]



        return alpha,color

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
    h = 256
    w = 256
    H = 256
    W = 256
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

    with open("archive/4d/poses/running_mean.pkl","rb") as f:
        running_pose = pickle.load(f)
        running_pose = torch.tensor(running_pose).cuda().float()
        running_pose = rc.axis_angle_to_matrix(running_pose.view(-1,3)).view(1, 55, 3, 3)
        # running_pose = torch.randn(1, 55, 3, 3).cuda()
    renderer = Renderer()
    smplx_layer = SMPLXLayer(model_path="./data/smplx/SMPLX_NEUTRAL_2020.npz").cuda()
    smplx_params = torch.nn.Parameter(running_pose[:,1:22,:,:], requires_grad=True)
    # smplx_params[15]["lr"] = 0.01
    dummy_smplx_output = smplx_layer()
    texture = torch.nn.Parameter(torch.ones_like(dummy_smplx_output.vertices[0]).cuda(),requires_grad=True)
    optimizer = torch.optim.AdamW([smplx_params,texture],lr=0.001)
    dataset = ViewDataset(CameraConfig, "cuda", type="test", size=4)
    faces_tensor = torch.from_numpy(smplx_layer.faces.astype("int32")).int().cuda()
    data = dataset[0]

    guidance = StableDiffusion("cuda", False, False)

    prompt = ["A person is walking "]

    text_embeds = guidance.get_text_embeds(prompt)
    empty_embeds = guidance.get_text_embeds(
        ["low motion, static statue, not moving, no motion"]
    )
    light_d = (data["rays_o"][0] + torch.randn(3, device=data["rays_o"].device, dtype=torch.float))
    light_d = safe_normalize(light_d)

    extern_data = np.load("./data/init_body/data.npz")
    vt = torch.tensor(extern_data["vt"]).cuda()
    ft = torch.tensor(extern_data["ft"]).cuda()
    albedo_image = cv2.imread("data/mesh_albedo.png")
    albedo_image = cv2.cvtColor(albedo_image, cv2.COLOR_BGR2RGB)
    albedo_image = albedo_image.astype(np.float32) / 255.0
    raw_albedo = torch.as_tensor(albedo_image, dtype=torch.float32, device="cuda")

    for i in range(1000):

        # First get the SMPL-X output
        smplx_output = smplx_layer(
            body_pose=smplx_params
        )

        # First, let's add some debug prints to understand our transformations
        print("Original vertices shape:", smplx_output.vertices[0].shape)
        print("Original joints shape:", smplx_output.joints[0].shape)
        # Normalize vertices and joints
        vertices_normalized, center, scale = normalize_vert(smplx_output.vertices[0], return_cs=True)

        print("Normalization center:", center)
        print("Normalization scale:", scale)
        joints = smplx_output.joints[0]
        joints_normalized = (joints - center) * scale

        # Debug prints for intermediate values
        # Debug prints
        print("Vertices center:", center)
        print("Vertices scale:", scale)
        print("Normalized vertices range:", vertices_normalized.min().item(), vertices_normalized.max().item())
        print("Normalized joints range:", joints_normalized.min().item(), joints_normalized.max().item())

        # Apply MVP transformation
        # Make sure the padding and expansion match exactly
        joints_homo = F.pad(joints_normalized, pad=(0, 1), mode="constant", value=1.0)
        joints_clip = torch.bmm(
            joints_homo.unsqueeze(0),
            torch.transpose(data["mvp"][None,...], 1, 2),
        ).float()

        print("Joints clip shape:", joints_clip.shape)
        print("Joints clip range:", joints_clip[..., :3].min().item(), joints_clip[..., :3].max().item())

        # Convert to NDC space with careful handling of the perspective divide
        eps = 1e-6  # Small epsilon to prevent division by zero
        w = joints_clip[..., 3:].clamp(min=eps)
        joints_ndc = joints_clip[..., :3] / w

        # Convert to NDC space
        joints_ndc = joints_clip[..., :3] / joints_clip[..., 3:]

        # Convert to pixel coordinates with proper bounds checking
        res = 256
        joints_2d = torch.zeros((joints_ndc.shape[1], 2))
        joints_2d[:, 0] = ((joints_ndc[0, :, 0] + 1.0) * res * 0.5).clamp(0, res-1)
        joints_2d[:, 1] = ((joints_ndc[0, :, 1] + 1.0) * res * 0.5).clamp(0, res-1)

        # Print the final 2D coordinates range
        print("2D joints range x:", joints_2d[:, 0].min().item(), joints_2d[:, 0].max().item())
        print("2D joints range y:", joints_2d[:, 1].min().item(), joints_2d[:, 1].max().item())

        # Render the mesh
        mesh = Mesh(vertices_normalized, faces_tensor,vt=vt,ft=ft)
        mesh.auto_normal()
        mesh.set_albedo(raw_albedo)
        color,normal,alpha = renderer(mesh, data["mvp"][None,...], light_d=light_d,h=256,w=256)
        color = color[:, :, :, :3].permute(0, 3, 1, 2)

        # Visualization with additional debug info
        plt.figure(figsize=(10, 10))
        plt.imshow(color[0].permute(1,2,0).detach().cpu().numpy())

        # Draw joints with different colors based on depth
        depths = joints_ndc[0, :, 2].detach().cpu().numpy()
        plt.scatter(joints_2d[:,0].detach().cpu(), 
            joints_2d[:,1].detach().cpu(),
            c=depths,
            cmap='jet',
            s=20)

        # Add joint numbers
        for i, (x, y) in enumerate(joints_2d.detach().cpu().numpy()):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

        # Add a colorbar to show depth
        plt.colorbar(label='Depth')

        plt.axis('off')
        plt.savefig("joints_with_nvdiffrast.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        def render_mesh_2():
            vertices = smplx_output.vertices[0].detach().cpu().numpy().squeeze()
            joints = smplx_output.joints[0].detach().cpu().numpy().squeeze()
            camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0,aspectRatio=1.0)
            camera_pose = np.eye(4)

            camera_translation = np.array([0, 0, 2.0])  # Default camera position
            camera_pose[:3, 3] = camera_translation
            def look_at(eye, target, up):
                """
                Create look-at rotation matrix
                Args:
                    eye: Camera position
                    target: Point to look at
                    up: Up vector
                Returns:
                    3x3 rotation matrix
                """
                z_axis = eye - target
                z_axis = z_axis / np.linalg.norm(z_axis)

                x_axis = np.cross(up, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)

                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)

                R = np.stack([x_axis, y_axis, z_axis], axis=1)
                return R
            camera_pose[:3,:3] = look_at(camera_translation,np.zeros(3),np.array([0,1,0]))

            def create_scene(vertices,joints,camera=None,camera_pose=None):
                """Create a scene with mesh and joints"""
                scene = pyrender.Scene(
                    bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3]
                )

                # Create mesh
                mesh = trimesh.Trimesh(
                    vertices=vertices, faces=smplx_layer.faces, process=False
                )
                mesh = pyrender.Mesh.from_trimesh(mesh)
                scene.add(mesh, "mesh")

                # Create joints visualization
                for joint in joints:
                    sm = trimesh.creation.uv_sphere(radius=0.01)
                    sm.visual.vertex_colors = [1.0, 0.0, 0.0]  # Red color
                    tfs = np.tile(np.eye(4), (1, 1, 1))
                    tfs[0, :3, 3] = joint
                    joints_mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                    scene.add(joints_mesh, "joints")

                # Add camera
                if camera is not None and camera_pose is not None:
                    scene.add(camera, pose=camera_pose)

                # Add light
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
                light_pose = np.eye(4)
                light_pose[:3, 3] = [0, -1, 1]
                scene.add(light, pose=light_pose)

                return scene

            scene = create_scene(vertices,joints,camera,camera_pose)

            pyrenderder = pyrender.OffscreenRenderer(256, 256,point_size=2.0)
            color, depth = pyrenderder.render(scene)

            def get_projection_matrix(
                fov=np.pi / 3, aspect_ratio=1.0, near=0.1, far=1000.0
            ):
                """
                Create perspective projection matrix
                Args:
                    fov: Field of view in radians
                    aspect_ratio: Aspect ratio of the image
                    near: Near clipping plane
                    far: Far clipping plane
                Returns:
                    4x4 projection matrix
                """
                f = 1.0 / np.tan(fov / 2)
                projection_matrix = np.zeros((4, 4))

                projection_matrix[0, 0] = f / aspect_ratio
                projection_matrix[1, 1] = f
                projection_matrix[2, 2] = (far + near) / (near - far)
                projection_matrix[2, 3] = 2 * far * near / (near - far)
                projection_matrix[3, 2] = -1.0

                return projection_matrix

            proj_matrix = get_projection_matrix()
            joints_homogeneous = np.hstack((joints, np.ones((joints.shape[0], 1))))
            world_to_camera = np.linalg.inv(camera_pose)
            joints_camera = (world_to_camera @ joints_homogeneous.T).T

            print(f"\nJoints in camera space (first 3):\n{joints_camera[:3]}")

            # Get normalized device coordinates (NDC)
            fov = np.pi / 3.0
            aspect = 1.0
            near = 0.1
            far = 1000.0

            # Perspective projection matrix
            f = 1.0 / np.tan(fov / 2)
            proj_matrix = np.array([
                [f/aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
                [0, 0, -1, 0]
            ])

            # Project to clip space
            joints_clip = (proj_matrix @ joints_camera.T).T
            print(f"\nJoints in clip space (first 3):\n{joints_clip[:3]}")

            # Perspective division
            w = joints_clip[:, 3:]
            joints_ndc = joints_clip[:, :3] / w
            print(f"\nJoints in NDC space (first 3):\n{joints_ndc[:3]}")

            # Convert to screen space (pixel coordinates)
            joints_2d = np.zeros((joints_ndc.shape[0], 2))
            joints_2d[:, 0] = (joints_ndc[:, 0] + 1.0) * 256 * 0.5
            joints_2d[:, 1] = (1.0 - (joints_ndc[:, 1] + 1.0) * 0.5) * 256

            print(f"\nFinal 2D joint positions (first 3):\n{joints_2d[:3]}")

            def visualize(color,joints_2d,show_joints=True):
                plt.figure(figsize=(10, 10))
                plt.imshow(color)
                if show_joints:
                    plt.scatter(joints_2d[:, 0], joints_2d[:, 1], c="r", s=20)
                plt.axis("off")
                plt.savefig("joints_pyrender.png")
                plt.close()
            visualize(color,joints_2d)

        render_mesh_2()

        def extract_keypoints_and_plot():

            def load_keypoint_rcnn():
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
                return DefaultPredictor(cfg)
            predictor = load_keypoint_rcnn()

            # Extract 2D keypoints
            def extract_keypoints(image, predictor):
                outputs = predictor(image)
                keypoints = outputs["instances"].pred_keypoints[0].cpu().numpy()
                return keypoints  

            keypoints = extract_keypoints(color[0].permute(1,2,0).detach().cpu().numpy(), predictor)

            def visualize_keypoints(image, keypoints, output_path=None):
                # Make a copy to draw on
                vis_image = image.copy()
                # Draw keypoints
                for idx, (x, y,z) in enumerate(keypoints):
                    # Convert to integers for cv2
                    x, y = int(x), int(y)

                    # Draw circle for keypoint
                    cv2.circle(vis_image, (x, y), 4, (0, 255, 0), -1)

                    # Add keypoint number
                    cv2.putText(
                        vis_image,
                        str(idx),
                        (x + 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

                if output_path:
                    cv2.imwrite(output_path, vis_image)
                else:
                    # Display image
                    cv2.imshow("Keypoints", vis_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                return vis_image

            visualize_keypoints(
                color[0].permute(1, 2, 0).detach().cpu().numpy(),
                keypoints,
                output_path="keypoints_2.png",
            )

        extract_keypoints_and_plot()
        image_latent = guidance.encode_imgs(color)

        dir_text_z = [
            empty_embeds[0],
            text_embeds[0],
        ]
        dir_text_z = torch.stack(dir_text_z)
        optimizer.zero_grad()
        loss = guidance.train_step(dir_text_z, image_latent , 100, True)
        print(loss.item())

        loss.backward()
        optimizer.step()
        break
    print("Test passed")
