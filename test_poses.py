import torch
import smplx,trimesh,pyrender
import numpy as np
from lib.rotation_conversions import matrix_to_axis_angle,axis_angle_to_matrix
from smplx.body_models import SMPLXLayer
import math
body_model_layer = SMPLXLayer(
    model_path="./data/smplx/SMPLX_NEUTRAL_2020.npz",
    num_betas=300,
    num_expression_coeffs=100,
    num_pca_comps=12,
    dtype=torch.float32,
    batch_size=1,
).cuda()

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

# Initialize your pose
decode_pose = (
    torch.eye(3, device="cuda", dtype=torch.float32)
    .view(1, 1, 3, 3)
    .expand(1, 21, -1, -1)
    .contiguous()
)
aa = np.load("/home/paulj/projects/TADA/4d/poses/E2_-__Jab_right_stageii.npz")["poses"][
    :, 3:66
]
aa = torch.from_numpy(aa).cuda().float()
decode_pose_full = axis_angle_to_matrix(aa.view(-1,3)).view(-1, 21, 3, 3)


for i in range(100,aa.shape[0]):
    decode_pose = decode_pose_full[None,i]


    # Define a 90 degree rotation matrix around the z-axis
    angle = math.pi / 2  # 90 degrees in radians
    rotation_matrix = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    ).cuda()

    # # Specify the joint index you want to rotate, e.g., joint 0
    # joint_index = 15

    # # Apply the rotation matrix to the specified joint
    # decode_pose[0, joint_index] = rotation_matrix


    output = body_model_layer(return_verts=True, body_pose=decode_pose)

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
