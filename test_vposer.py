from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
import pickle
import torch
import pyrender
import smplx
vp , ps = load_model('/home/paulj/projects/TADA/V02_05', model_code=VPoser, remove_words_in_model_weights='vp_model.',disable_grad=True)

vp = vp.cuda()

motion_file = pickle.load(open('/home/paulj/projects/TADA/4d/poses/running.pkl', 'rb'))
motion = torch.from_numpy(motion_file['body_pose'][0]).float().cuda().unsqueeze(0)
print(motion.shape)
body_z = vp.encode(motion).mean



r = pyrender.OffscreenRenderer(viewport_width=400,viewport_height=400,point_size=1.0)

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
            ).to(self.device)

curr_rot = Rot.from_euler("zyx", [0, 0, 0])
transform33 = curr_rot.as_matrix()
transform = np.eye(4)
transform[:3,:3] = transform33
transform[2,3] = -1
fuze_trimesh.apply_transform(transform)

scene = pyrender.Scene.from_trimesh_scene(fuze_trimesh)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
s = np.sqrt(2)/2
camera_pose = np.array([
        [1.0, 0,   0,  0],
        [0,  1.0, 0.0, 0],
        [0.0,  0,   1,   1],
        [0.0,  0.0, 0.0, 1.0],])
# camera_pose[:3,:3] = curr_rot.as_matrix()
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=5.0,
                            innerConeAngle=np.pi/6.0,
                            outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
color, depth = r.render(scene)
plt.imshow(color)
print(transform33)
print(transform)

breakpoint()