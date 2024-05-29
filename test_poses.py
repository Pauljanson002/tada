import torch
import smplx,trimesh,pyrender
import numpy as np
from lib.rotation_conversions import matrix_to_axis_angle
value = torch.load("boxing_start.pt")

body_model = smplx.create(
    model_path="./data/smplx/SMPLX_NEUTRAL_2020.npz",
    model_type="smplx",
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
).cuda()


decode_pose = matrix_to_axis_angle(value.view(-1, 3, 3))
left_hand_pose = torch.zeros(15,3).cuda()
right_hand_pose = torch.zeros(15,3).cuda()
left_hand_pose[0,:] = decode_pose[None,21,:]
right_hand_pose[0,:] = decode_pose[None,22,:]

decode_pose = decode_pose[None,:21,:].view(1, -1).cuda()


output = body_model(return_verts=True, body_pose=decode_pose,left_hand_pose=left_hand_pose,right_hand_pose=right_hand_pose)

mesh = trimesh.Trimesh(output.vertices[0].cpu().detach().numpy(), body_model.faces)

mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))




scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
render_mesh = pyrender.Mesh.from_trimesh(
    mesh,
    smooth=True,
    material=pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        roughnessFactor=0.6,
        alphaMode="OPAQUE",
        baseColorFactor=(0.3, 0.5, 0.55, 1.0),
    ),
)

scene.add(render_mesh)

pyrender.Viewer(scene, use_raymond_lighting=True)

