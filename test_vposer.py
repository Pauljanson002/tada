from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
import pickle
import torch
import pyrender
import trimesh
import smplx
import numpy as np
vp , ps = load_model('/home/paulj/projects/TADA/V02_05', model_code=VPoser, remove_words_in_model_weights='vp_model.',disable_grad=True)

vp = vp.cuda()

motion_file = pickle.load(open('/home/paulj/projects/TADA/4d/poses/diving.pkl', 'rb'))
motion = torch.from_numpy(motion_file['body_pose'][0]).float().cuda().unsqueeze(0)
print(motion.shape)
# body_z = vp.encode(motion).mean
body_z = torch.randn(1, 32).cuda()
decode_pose = vp.decode(body_z)["pose_body"].contiguous().view(-1, 63)

print("Reconstruction loss: ", torch.nn.functional.mse_loss(motion, decode_pose).item())
body_model = smplx.create(
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
            ).cuda()


output = body_model(return_verts=True, body_pose=decode_pose)
output_2 = body_model(return_verts=True, body_pose=motion)
output_2.vertices[0] += torch.tensor([0.5, 0, 0]).cuda()
mesh = trimesh.Trimesh(output.vertices[0].cpu().detach().numpy(), body_model.faces)

mesh.apply_transform(trimesh.transformations.rotation_matrix(
    np.radians(90), [1, 0, 0]))

mesh_2 = trimesh.Trimesh(output_2.vertices[0].cpu().detach().numpy(), body_model.faces)
mesh_2.apply_transform(trimesh.transformations.rotation_matrix(
    np.radians(90), [1, 0, 0]))
mesh_2.vertices += np.array([-0.5, 0, 0])


scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.3, 0.3, 0.3))
render_mesh = pyrender.Mesh.from_trimesh(
    mesh,
    smooth=True,
    material=pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        roughnessFactor=0.6,
        alphaMode='OPAQUE',
        baseColorFactor=(0.3, 0.5, 0.55, 1.0)
))

render_mesh_2 = pyrender.Mesh.from_trimesh(
    mesh_2,
    smooth=True,
    material=pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        roughnessFactor=0.6,
        alphaMode='OPAQUE',
        baseColorFactor=(0.0, 0.0, 1.0, 1.0)
))
scene.add(render_mesh)
scene.add(render_mesh_2)

pyrender.Viewer(scene, use_raymond_lighting=True)

