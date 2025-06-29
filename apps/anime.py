import os
import json
import pickle as pkl
import random


os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # Uncommnet this line while running remotely

import argparse
import cv2
import torch
import smplx
import trimesh
import imageio
import pyrender
from PIL import Image
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

from lib.common.obj import Mesh
from lib.common.remesh import subdivide_inorder
from lib.common.utils import SMPLXSeg
from lib.common.lbs import warp_points
from lib.common.obj import normalize_vert, compute_normal
import joblib
from PIL import Image
def get_motion_diffusion_pose(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    x_translations = data['motion'][-1, :3]
    motion = data['motion'][:-1]

    motion = torch.as_tensor(motion).permute(2, 0, 1)
    x_translations = torch.as_tensor(x_translations).permute(1, 0)
    from lib.common.rotation_conversions import rotation_6d_to_matrix, matrix_to_euler_angles, matrix_to_axis_angle
    rotations = rotation_6d_to_matrix(motion) # [540, 24, 3, 3]
    rotations = matrix_to_axis_angle(rotations.reshape(-1, 3, 3)).reshape(-1, 24, 3)

    return rotations, x_translations


def build_new_mesh(v, f, vt, ft):
    # build a correspondences dictionary from the original mesh indices to the (possibly multiple) texture map indices
    f_flat = f.flatten()
    ft_flat = ft.flatten()
    correspondences = {}

    # traverse and find the corresponding indices in f and ft
    for i in range(len(f_flat)):
        if f_flat[i] not in correspondences:
            correspondences[f_flat[i]] = [ft_flat[i]]
        else:
            if ft_flat[i] not in correspondences[f_flat[i]]:
                correspondences[f_flat[i]].append(ft_flat[i])

    # build a mesh using the texture map vertices
    new_v = np.zeros((v.shape[0], vt.shape[0], 3))
    for old_index, new_indices in correspondences.items():
        for new_index in new_indices:
            new_v[:, new_index] = v[:, old_index]

    # define new faces using the texture map faces
    f_new = ft
    return new_v, f_new


def render_mesh_helper(mesh, input_renderer, z_offset=1.0, xmag=0.8, y=1.2, z=1, use_default_color=True):
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0]))
    render_mesh = pyrender.Mesh.from_trimesh(
        mesh,
        smooth=True,
        material=pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            alphaMode='OPAQUE',
            baseColorFactor=(0.3, 0.5, 0.55, 1.0)
        ) if use_default_color else None
    )

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3), bg_color=[1.0, 1.0, 1.0, 0])

    ymag = xmag * z_offset
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0.7, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, y],  # 0.25
                            [0, 0, 1, z],  # 0.2
                            [0, 0, 0, 1]])

    if True:
        light_color = np.array([1., 1., 1.])
        light = pyrender.DirectionalLight(color=light_color, intensity=2)

        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, 0.7, 2.0])
        scene.add(light, pose=light_pose.copy())
    else:
        light = pyrender.PointLight(color=np.array([1.0, 1.0, 1.0]) * 0.2, intensity=2)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose)

        spot_l = pyrender.SpotLight(color=np.ones(3), intensity=15.0,
                                    innerConeAngle=np.pi / 3, outerConeAngle=np.pi / 2)

        light_pose[:3, 3] = [-1, 2, 2]
        scene.add(spot_l, pose=light_pose)

        light_pose[:3, 3] = [1, 2, 2]
        scene.add(spot_l, pose=light_pose)

    # flags = pyrender.RenderFlags.SKIP_CULL_FACES
    flags = pyrender.RenderFlags.RGBA
    color, alpha = input_renderer.render(scene, flags=flags)
    # print(alpha.shape, alpha.max())
    # exit()

    return color.astype(np.uint8)


def render_mesh_helper_2(mesh, input_renderer, z_offset=1.0, xmag=0.8, y=1.2, z=1, use_default_color=True):
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0]))
    render_mesh = pyrender.Mesh.from_trimesh(
        mesh,
        smooth=True,
        material=pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            alphaMode='OPAQUE',
            baseColorFactor=(0.3, 0.5, 0.55, 1.0)
        ) if use_default_color else None
    )

    scene.add(render_mesh)

    camera_pose = np.eye(4)
    camera = pyrender.IntrinsicsCamera(fx=5000, fy=5000, cx=640, cy=640,zfar=1000)
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)
    # camera_pose[:3, 3] = np.array([0, 0.7, 1.0 - z_offset])
    # scene.add(camera, pose=[[1, 0, 0, 0],
    #                         [0, 1, 0, y],  # 0.25
    #                         [0, 0, 1, z],  # 0.2
    #                         [0, 0, 0, 1]])
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    scene.add(dl)

    color, alpha = input_renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    # print(alpha.shape, alpha.max())
    # exit()
    return color.astype(np.uint8), alpha.astype(np.uint8)

def render_mesh_helper_aist(mesh, input_renderer, z_offset=0.5, xmag=0.5, use_default_color=True):
    render_mesh = pyrender.Mesh.from_trimesh(
        mesh,
        smooth=True,
        material=pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            alphaMode='OPAQUE',
            baseColorFactor=(0.3, 0.5, 0.55, 1.0)
        ) if use_default_color else None
    )

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3), bg_color=[1.0, 1.0, 1.0, 0])

    # ymag = xmag * z_offset
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=xmag)
    scene.add(render_mesh, pose=np.eye(4))

    pose = np.eye(4)
    scene.add(camera, pose=np.eye(4))

    # light_color = np.array([1., 1., 1.])
    # light = pyrender.DirectionalLight(color=light_color, intensity=0.001)

    # light_pose = np.eye(4)
    # light_pose[:3, 3] = np.array([0, 0.7, 2.0])
    # scene.add(light, pose=light_pose.copy())

    # flags = pyrender.RenderFlags.SKIP_CULL_FACES
    flags = pyrender.RenderFlags.RGBA
    color, _ = input_renderer.render(scene, flags=flags)

    return color.astype(np.uint8)


class Animation:
    def __init__(self, render_res=512):
        self.device = torch.device("cuda")

        # load data
        init_data = np.load('./data/init_body/data.npz')
        self.dense_faces = torch.as_tensor(init_data['dense_faces'], device=self.device)
        self.dense_lbs_weights = torch.as_tensor(init_data['dense_lbs_weights'], device=self.device)
        self.unique = init_data['unique']
        self.vt = init_data['vt']
        self.ft = init_data['ft']

        model_params = dict(
            model_path="./data/smplx/SMPLX_NEUTRAL_2020.npz",
            model_type='smplx',
            use_pca=False,
            flat_hand_mean=False,
            num_betas=300,
            num_expression_coeffs=100,
            num_pca_comps=12,
            dtype=torch.float32,
            batch_size=1,
        )
        self.body_model = smplx.create(**model_params).to(device='cuda')
        self.smplx_face = self.body_model.faces.astype(np.int32)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=render_res, viewport_height=render_res,
                                                   point_size=1.0)
        # self.renderer = pyrender.Renderer(viewport_width=render_res, viewport_height=render_res,)

    def load_ckpt_data(self, ckpt_file,subject):
        # model_data = torch.load(ckpt_file)["model"]
        model_data = torch.load(ckpt_file)
        if subject in ["fatguy","shortguy","thinguy","tallguy","athleticguy"]:
            model_data = model_data["model"]
        self.expression = model_data["expression"] if "expression" in model_data else None
        self.jaw_pose = model_data["jaw_pose"] if "jaw_pose" in model_data else None
        # self.raw_albedo = torch.sigmoid(model_data['raw_albedo'])
        self.betas = model_data['betas']
        self.v_offsets = model_data['v_offsets']
        self.v_offsets[SMPLXSeg.eyeball_ids] = 0.
        self.v_offsets[SMPLXSeg.hands_ids] = 0.

        # tex to trimesh texture
        vt = self.vt.copy()
        vt[:, 1] = 1 - vt[:, 1]
        # albedo = T.ToPILImage()(self.raw_albedo.permute(2, 0, 1))
        albedo = Image.open(f"characters/{subject}/mesh_albedo.png")
        self.trimesh_visual = trimesh.visual.TextureVisuals(
            uv=vt,
            image=albedo,
            material=trimesh.visual.texture.SimpleMaterial(
                image=albedo,
                diffuse=[255, 255, 255, 255],
                ambient=[255, 255, 255, 255],
                specular=[0, 0, 0, 255],
                glossiness=0)
        )

    def forward_talkshow(self, pose_file, video_save_path, interval=5):
        self.v_offsets[SMPLXSeg.lips_ids] = 0
        smplx_params = np.load(pose_file)
        scan_v_posed = []
        smplx_v_posed = []
        smplx_v_canon = []

        for i in tqdm(range(0, smplx_params.shape[0], interval)):
            params_batch = torch.as_tensor(smplx_params[i:i + 1], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                output = self.body_model(
                    betas=self.betas,
                    jaw_pose=params_batch[:, 0:3],
                    global_orient=params_batch[:, 9:12],
                    body_pose=params_batch[:, 12:75].view(-1, 21, 3),
                    left_hand_pose=params_batch[:, 75:120],
                    right_hand_pose=params_batch[:, 120:165],
                    expression=params_batch[:, 165:265],
                    return_verts=True
                )

            v_cano = output.v_posed[0]
            # re-mesh
            v_cano_dense = subdivide_inorder(v_cano, self.smplx_face[SMPLXSeg.remesh_mask], self.unique).squeeze(0)
            # add offsets
            vn = compute_normal(v_cano_dense, self.dense_faces)[0]
            v_cano_dense += self.v_offsets * vn
            # do LBS
            v_posed_dense = warp_points(v_cano_dense, self.dense_lbs_weights, output.joints_transform[:, :55])

            smplx_v_posed.append(output.vertices)
            scan_v_posed.append(v_posed_dense)
            smplx_v_canon.append(output.vertices)

        scan_v_posed = torch.cat(scan_v_posed).detach().cpu().numpy()
        smplx_v_posed = torch.cat(smplx_v_posed).detach().cpu().numpy()
        smplx_v_canon = torch.cat(smplx_v_canon).detach().cpu().numpy()

        new_scan_v_posed, new_face = build_new_mesh(scan_v_posed, self.dense_faces, self.vt, self.ft)

        out_frames = []
        for idx in tqdm(range(0, scan_v_posed.shape[0])):
            mesh = trimesh.Trimesh(new_scan_v_posed[idx], new_face, visual=self.trimesh_visual, process=False)
            mesh_tex = render_mesh_helper(mesh, self.renderer, use_default_color=False)
            out_frames.append(np.hstack([mesh_tex]))

        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        imageio.mimsave(video_save_path, out_frames, fps=30 // interval, quality=8, macro_block_size=1)
        print("save to", video_save_path)


    def forward_disk(self,args,video_save_path="/home/paulj/projects/TADA/save_dir/video.mp4"):
        self.v_offsets[SMPLXSeg.lips_ids] = 0
        # smplx_params = np.load(pose_file)
        scan_v_posed = []
        smplx_v_posed = []
        smplx_v_canon = []
        pickle_file = pkl.load(open(f"archive/4d_archive/{args.motion}/poses/merged.pkl","rb"))
        # results_file = pkl.load(open(f"/home/paulj/projects/collosal/4D-Humans/outputs/results/demo_diving.pkl","rb"))
        for i in tqdm(range(0, len(pickle_file["body_pose"])),disable=True):
            
            # params_batch = torch.as_tensor(smplx_params[i:i + 1], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                jaw_pose=torch.tensor(pickle_file['jaw_pose'][i], dtype=torch.float32, device=self.device).unsqueeze(0)
                global_orient=torch.tensor(pickle_file['global_orient'][i], dtype=torch.float32, device=self.device).unsqueeze(0)
                body_pose=torch.tensor(pickle_file["body_pose"][i], dtype=torch.float32, device=self.device).unsqueeze(0).view(-1, 21, 3)
                left_hand_pose=torch.tensor(pickle_file['left_hand_pose'][i], dtype=torch.float32, device=self.device).unsqueeze(0)
                right_hand_pose=torch.tensor(pickle_file['right_hand_pose'][i], dtype=torch.float32, device=self.device).unsqueeze(0)
                expression=torch.tensor(pickle_file['expression'][i], dtype=torch.float32, device=self.device).unsqueeze(0)

                output = self.body_model(
                    betas=self.betas,
                    jaw_pose=jaw_pose,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    # expression=expression,
                    return_verts=True,
                )

            v_cano = output.v_posed[0]
            # re-mesh
            v_cano_dense = subdivide_inorder(v_cano, self.smplx_face[SMPLXSeg.remesh_mask], self.unique).squeeze(0)
            # add offsets
            vn = compute_normal(v_cano_dense, self.dense_faces)[0]
            v_cano_dense += self.v_offsets * vn
            # do LBS
            v_posed_dense = warp_points(v_cano_dense, self.dense_lbs_weights, output.joints_transform[:, :55])
            pred_cam_t = torch.load(f"archive/4d_archive/{args.motion}/pred_cam/pred_cam_t_{i}.pt").to(self.device)
            # pred_cam_t = self.camera_params[list(self.camera_params.keys())[i]]["camera"][0]
            # pred_cam_t = torch.tensor(pred_cam_t, device=self.device).unsqueeze(0)
            pred_cam_t_bs = pred_cam_t.unsqueeze(1).repeat(1, v_posed_dense.size(1), 1)
            v_posed_dense += pred_cam_t_bs
            smplx_v_posed.append(output.vertices)
            scan_v_posed.append(v_posed_dense)
            smplx_v_canon.append(output.vertices)


        scan_v_posed = torch.cat(scan_v_posed).detach().cpu().numpy()
        smplx_v_posed = torch.cat(smplx_v_posed).detach().cpu().numpy()
        smplx_v_canon = torch.cat(smplx_v_canon).detach().cpu().numpy()

        new_scan_v_posed, new_face = build_new_mesh(scan_v_posed, self.dense_faces, self.vt, self.ft)

        out_frames = []
        for idx in tqdm(range(0, scan_v_posed.shape[0])):
            mesh = trimesh.Trimesh(new_scan_v_posed[idx], new_face, visual=self.trimesh_visual, process=False)
            mesh_tex,mask = render_mesh_helper_2(mesh, self.renderer, use_default_color=False)
            background = Image.open(f"archive/4d_archive/{args.motion}/backgrounds/{idx}.png")
            # mask = mesh_tex[:, :, 3:4]
            # make mask to 0 to 1
           # mask = mask / 255
            mask = mask/17
            mask = mask[:,:,None]
            mesh_image = mesh_tex[:, :, :3]
            final_image = background * (1 - mask) + mesh_image * mask
            final_image = final_image.astype(np.uint8)
            #out_frames.append(np.hstack([mesh_tex]))
            out_frames.append(np.hstack([final_image]))
            # save image 


        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        imageio.mimsave(video_save_path, out_frames, fps=30, quality=8, macro_block_size=1)
        print("save to", video_save_path)

    def forward_aist(self, video_save_path, aist_dir="../data/aist", interval=5):
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

        mapping = list(open(f"{aist_dir}/cameras/mapping.txt", 'r').read().splitlines())
        motion_setting_dict = {}
        for pairs in mapping:
            motion, setting = pairs.split(" ")
            motion_setting_dict[motion] = setting
        # motion_name = random.choice(os.listdir(f"{aist_dir}/motion/"))
        motion_name = "gHO_sBM_cAll_d19_mHO0_ch04.pkl"

        # load camera data
        setting = motion_setting_dict[motion_name[:-4]]
        camera_path = open(f"{aist_dir}/cameras/{setting}.json", 'r')
        camera_params = json.load(camera_path)[0]
        rvec = np.array(camera_params['rotation'])
        tvec = np.array(camera_params['translation'])
        matrix = np.array(camera_params['matrix']).reshape((3, 3))
        distortions = np.array(camera_params['distortions'])

        # load motion
        smpl_data = pkl.load(open(f"{aist_dir}/motion/{motion_name}", 'rb'))
        poses = smpl_data['smpl_poses']  # (N, 24, 3)
        scale = smpl_data['smpl_scaling']  # (1,)
        trans = smpl_data['smpl_trans']  # (N, 3)
        poses = torch.from_numpy(poses).view(-1, 24, 3).float()
        # interval = poses.shape[0] // 400
        poses = poses[::interval]
        trans = trans[::interval]
        print("NUM pose", poses.shape, trans.shape)

        scan_v_posed = []
        for i, (pose, t) in tqdm(enumerate(zip(poses, trans))):
            body_pose = torch.as_tensor(pose[None, 1:22].view(1, 21, 3), device=self.device)
            global_orient = torch.as_tensor(pose[None, :1], device=self.device)
            output = self.body_model(
                betas=self.betas,
                global_orient=global_orient,
                jaw_pose=self.jaw_pose,
                body_pose=body_pose,
                expression=self.expression,
                return_verts=True)
            v_cano = output.v_posed[0]
            # re-mesh
            v_cano_dense = subdivide_inorder(v_cano, self.smplx_face[SMPLXSeg.remesh_mask], self.unique).squeeze(0)
            # add offsets
            vn = compute_normal(v_cano_dense, self.dense_faces)[0]
            v_cano_dense += self.v_offsets * vn
            # do LBS
            v_posed_dense = warp_points(v_cano_dense, self.dense_lbs_weights, output.joints_transform[:, :55])
            scan_v_posed.append(v_posed_dense)

        scan_v_posed = torch.cat(scan_v_posed).detach().cpu().numpy()
        new_scan_v_posed, new_face = build_new_mesh(scan_v_posed, self.dense_faces, self.vt, self.ft)

        out_frames = []
        for i, (posed_points, t) in tqdm(enumerate(zip(new_scan_v_posed, trans))):
            posed_points = posed_points * scale + t[None, :]

            pts2d = cv2.projectPoints(
                posed_points, rvec=rvec, tvec=tvec, cameraMatrix=matrix, distCoeffs=distortions)[0][:, 0]
            posed_points = np.concatenate([pts2d, posed_points[:, 2:]], axis=1)

            posed_points[:, 0] -= 420
            posed_points[:, 1] = 1080 - posed_points[:, 1]
            posed_points = posed_points / 1080 * 2 - 1

            posed_points[:, 1] += 0.35

            mesh = trimesh.Trimesh(posed_points, new_face, visual=self.trimesh_visual, process=False)
            mesh_tex = render_mesh_helper_aist(mesh, self.renderer, use_default_color=False)

            out_frames.append(np.hstack([mesh_tex]))
        video_save_path = video_save_path[:-4] + "_" + motion_name + ".mp4"
        imageio.mimsave(video_save_path, out_frames, fps=30 // interval, quality=8, macro_block_size=1)

    def forward_mdm(self, mdm_file_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        mdm_body_pose, translate = get_motion_diffusion_pose(mdm_file_path)
        translate = translate.to(self.device)

        for i, (pose, t) in tqdm(enumerate(zip(mdm_body_pose, translate))):
            body_pose = torch.as_tensor(pose[None, 1:22].view(1, 21, 3), device=self.device)
            global_orient = torch.as_tensor(pose[None, :1], device=self.device)
            output = self.body_model(
                betas=self.betas,
                global_orient=global_orient,
                jaw_pose=self.jaw_pose,
                body_pose=body_pose,
                expression=self.expression,
                return_verts=True
            )
            v_cano = output.v_posed[0]
            # re-mesh
            v_cano_dense = subdivide_inorder(v_cano, self.smplx_face[SMPLXSeg.remesh_mask], self.unique).squeeze(0)
            # add offsets
            vn = compute_normal(v_cano_dense, self.dense_faces)[0]
            v_cano_dense += self.v_offsets * vn
            # do LBS
            v_posed_dense = warp_points(v_cano_dense, self.dense_lbs_weights, output.joints_transform[:, :55])
            # translate
            v_posed_dense += t - translate[0]

            mesh = Mesh(v_posed_dense[0].detach(), self.dense_faces,
                        vt=torch.as_tensor(self.vt),
                        ft=torch.as_tensor(self.ft),
                        albedo=self.raw_albedo)
            mesh.auto_normal()
            mesh.write(f"{save_dir}/{i:03d}/mesh.obj")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject',default="Batman",type=str, required=False, help="subject's name or input text prompt")
    parser.add_argument("--motion", default="running", type=str, required=False, help="motion type")
    parser.add_argument('--render_res', default=1280, type=int, help="rendered image resolution")
    parser.add_argument('--save_dir', default='save_dir', type=str, help="save dir")
    parser.add_argument('--workspace', default='workspace', type=str, help="workspace dir")
    parser.add_argument('--motion_type', default="disk", type=str, help="aist, talkshow or mdm")
    parser.add_argument('--talkshow_file', default="data/talkshow/rich.npy", type=str,
                        help="file of talkshow file for face animation")
    parser.add_argument('--mdm_file', default="./data/mdm/sample00_rep00_smpl_params.npy", type=str,
                        help="file of motion diffusion file for face animation")
    args = parser.parse_args()

    # ckpt_file = f"{args.workspace}/tada/{args.subject}/checkpoints/tada_ep0150.pth"
    if args.subject in ["fatguy","shortguy","thinguy","tallguy","athleticguy"]:
        ckpt_file = f"characters/{args.subject}/wo_normal_supervision_ep0150.pth"
    else:
        ckpt_file = f"characters/{args.subject}/params.pt"
    assert os.path.exists(ckpt_file)
    animator = Animation(render_res=args.render_res)
    animator.load_ckpt_data(ckpt_file,args.subject)

    if args.motion_type == "talkshow" and os.path.exists(args.talkshow_file):
        save_path = f"{args.save_dir}/{args.subject}-talkshow.mp4"
        animator.forward_talkshow(args.talkshow_file, save_path, interval=10)
    elif args.motion_type == "aist":
        save_path = f"{args.save_dir}/{args.subject}-aist.mp4"
        animator.forward_aist(save_path)
    elif args.motion_type == "mdm":
        save_path = f"{args.save_dir}/{args.subject}-mdm/"
        animator.forward_mdm(args.mdm_file, save_path)
    elif args.motion_type == "disk":
        save_path = f"/home/paulj/projects/TADA/save_dir/{args.subject}-{args.motion}.mp4"
        animator.forward_disk(args,video_save_path=save_path)

