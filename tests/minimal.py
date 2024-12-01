import numpy as np
import torch
import smplx
import pyrender
import trimesh
import matplotlib.pyplot as plt
import nvdiffrast.torch as dr


class DualRenderer:
    def __init__(self, model_path, gender="neutral", img_size=512):
        # Initialize SMPLX model
        self.smplx_model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext="npz",
        ).cuda()

        self.img_size = img_size

        # PyRender renderer
        self.pyrenderer = pyrender.OffscreenRenderer(
            viewport_width=img_size, viewport_height=img_size, point_size=2.0
        )

        # NVDiffrast renderer
        try:
            self.nvrenderer = dr.RasterizeCudaContext()
        except:
            self.nvrenderer = dr.RasterizeGLContext()

    def normalize_vertices(self, vertices):
        vmax, vmin = vertices.max(0)[0], vertices.min(0)[0]
        center = (vmax + vmin) * 0.5
        scale = 1.0 / (vmax - vmin).max()
        return (vertices - center) * scale, center, scale

    def get_projection_matrix(self, fov=np.pi / 3):
        """Get perspective projection matrix"""
        aspect_ratio = 1.0
        near = 0.1
        far = 1000.0

        f = 1.0 / np.tan(fov / 2)
        return np.array(
            [
                [f / aspect_ratio, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0],
            ]
        )

    def render_both(self, body_pose, global_orient=None, transl=None, betas=None):
        # Set default parameters if not provided
        if betas is None:
            betas = torch.zeros(1, 10, device=body_pose.device)
        if global_orient is None:
            global_orient = torch.zeros(1, 3, device=body_pose.device)
        if transl is None:
            transl = torch.zeros(1, 3, device=body_pose.device)

        # Get SMPLX output
        output = self.smplx_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            return_verts=True,
        )

        vertices = output.vertices[0].detach()
        joints = output.joints[0].detach()

        # Normalize vertices and joints
        vertices_normalized, center, scale = self.normalize_vertices(vertices)
        joints_normalized = (joints - center) * scale

        # 1. PyRender rendering
        vertices_np = vertices_normalized.cpu().numpy()
        joints_np = joints_normalized.cpu().numpy()

        # Create camera and scene for PyRender
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        camera_pose = np.eye(4)
        camera_pose[2, 3] = 2.0

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])

        # Add mesh to scene
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=self.smplx_model.faces)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh, "mesh")

        # Add joints to scene
        for joint in joints_np:
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = [1.0, 0.0, 0.0]
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[0, :3, 3] = joint
            joints_mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_mesh)

        scene.add(camera, pose=camera_pose)

        # Add light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=camera_pose)

        # Render with PyRender
        py_color, py_depth = self.pyrenderer.render(scene)

        # 2. NVDiffrast rendering
        # Create MVP matrix
        proj_matrix = torch.tensor(
            self.get_projection_matrix(), device=vertices.device, dtype=torch.float32
        )

        # Apply MVP transformation to joints
        joints_homo = torch.nn.functional.pad(joints_normalized, (0, 1), value=1.0)
        joints_clip = torch.matmul(joints_homo, proj_matrix.T)

        # Perspective divide
        w = joints_clip[:, 3:].clamp(min=1e-7)
        joints_ndc = joints_clip[:, :3] / w

        # Convert to pixel coordinates
        joints_2d = torch.zeros((joints_ndc.shape[0], 2), device=joints_ndc.device)
        joints_2d[:, 0] = ((joints_ndc[:, 0] + 1.0) * self.img_size * 0.5).clamp(
            0, self.img_size - 1
        )
        joints_2d[:, 1] = ((1.0 - joints_ndc[:, 1]) * self.img_size * 0.5).clamp(
            0, self.img_size - 1
        )

        # Render with NVDiffrast
        vertices_homo = torch.nn.functional.pad(vertices_normalized, (0, 1), value=1.0)
        v_clip = torch.matmul(vertices_homo, proj_matrix.T).unsqueeze(0)

        faces_tensor = torch.tensor(self.smplx_model.faces, device=vertices.device, dtype=torch.int32)
        rast, _ = dr.rasterize(
            self.nvrenderer,
            v_clip,
            faces_tensor,
            (self.img_size, self.img_size),
        )

        # Create simple shading
        nv_color = torch.ones_like(vertices_normalized).unsqueeze(0)
        nv_color, _ = dr.interpolate(
            nv_color, rast, faces_tensor
        )

        return {
            "pyrender": {"color": py_color, "depth": py_depth},
            "nvdiffrast": {
                "color": nv_color[0].cpu().numpy(),
                "joints_2d": joints_2d.cpu().numpy(),
            },
        }

    def visualize_comparison(self, results):
        """Visualize both renderings side by side with joints"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # PyRender result
        ax1.imshow(results["pyrender"]["color"])
        ax1.set_title("PyRender")
        ax1.axis("off")

        # NVDiffrast result
        ax2.imshow(results["nvdiffrast"]["color"])
        ax2.scatter(
            results["nvdiffrast"]["joints_2d"][:, 0],
            results["nvdiffrast"]["joints_2d"][:, 1],
            c="r",
            s=20,
        )
        ax2.set_title("NVDiffrast")
        ax2.axis("off")

        plt.savefig("comparison.png")
        plt.close()


# Example usage
if __name__ == "__main__":
    model_path = "/home/paulj/projects/TADA/data/smplx/SMPLX_NEUTRAL_2020.npz"
    renderer = DualRenderer(model_path)

    # Create some test pose
    body_pose = torch.zeros(1, 63).cuda()

    # Render with both engines
    results = renderer.render_both(body_pose)

    # Visualize comparison
    renderer.visualize_comparison(results)
