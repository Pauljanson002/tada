import torch
import smplx
import trimesh
import pyrender
import fast_simplification
import nvdiffrast.torch as dr
import xatlas
class MyClass:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def forward(self):
        model_output = self.body_model(return_verts=True)  
        vertices = model_output.vertices[0].cpu().detach().numpy()
        faces = self.body_model.faces
        # simplification
        # vertices, faces = fast_simplification.simplify(vertices, faces, 0.9)
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces,)
        vmapping , indices , uvs =  xatlas.parametrize(mesh_trimesh.vertices, mesh_trimesh.faces)
        uvs = torch.tensor(uvs).to(self.device)
        indices = torch.tensor(indices.astype(int)).to(self.device).to(torch.int32)
        # self.glctx = dr.RasterizeCudaContext()
        # B = 1
        # mvp = torch.eye(4).unsqueeze(0).expand(B, -1, -1).to(self.device)
        # v_clip = torch.bmm(torch.cat([torch.tensor(vertices.astype(float)).float().to(self.device), torch.ones((vertices.shape[0], 1)).float().to(self.device)], dim=1).unsqueeze(0).expand(B, -1, -1), torch.transpose(mvp, 1, 2)).float()
        # res = (512, 512)
        # rast, rast_db = dr.rasterize(self.glctx, v_clip, torch.tensor(faces).int().to(self.device), res)
        # texc, texc_db = dr.interpolate(
        #     uvs[None, ...], rast, indices, rast_db=rast_db, diff_attrs="all"
        # )

        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh_trimesh,wireframe=False))
        pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == '__main__':
    my_class = MyClass()
    my_class.forward()
