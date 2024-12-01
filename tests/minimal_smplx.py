import numpy as np
import torch
import smplx
import pyrender
import trimesh
import matplotlib.pyplot as plt



class SMPLXRenderer:
    def __init__(self, model_path, gender="neutral", img_size=1024):
        """
        Initialize SMPLX model and renderer
        Args:
            model_path: Path to the SMPLX models
            gender: 'neutral', 'male' or 'female'
            img_size: Size of the output image (assumed square)
        """
        # Initialize SMPLX model
        self.smplx_model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext="npz",
        )

        self.img_size = img_size

        # Initialize renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_size, viewport_height=img_size, point_size=2.0
        )

    def get_projection_matrix(
        self, fov=np.pi / 3, aspect_ratio=1.0, near=0.1, far=1000.0
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

    def create_camera(self, camera_translation=None):
        """
        Create a perspective camera
        Args:
            camera_translation: Optional camera position
        Returns:
            pyrender.camera.PerspectiveCamera and its pose
        """
        if camera_translation is None:
            camera_translation = np.array([0, 0, 2.0])  # Default camera position

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

        # Create camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        # Rotate camera to look at origin
        camera_pose[:3, :3] = self.look_at(
            camera_translation, np.zeros(3), np.array([0, 1, 0])
        )

        return camera, camera_pose

    def look_at(self, eye, target, up):
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

    def create_scene(self, vertices, joints, camera=None, camera_pose=None):
        """Create a scene with mesh and joints"""
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3]
        )

        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=self.smplx_model.faces, process=False
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

    def render_mesh(self, betas=None, global_orient=None, body_pose=None, transl=None):
        """
        Render SMPLX mesh and joints
        """
        # Set default parameters if not provided
        if betas is None:
            betas = torch.zeros(1, 10)
        if global_orient is None:
            global_orient = torch.zeros(1, 3)
        if body_pose is None:
            body_pose = torch.zeros(1, 69)
        if transl is None:
            transl = torch.zeros(1, 3)

        print("Input shapes:")
        print(f"betas: {betas.shape}")
        print(f"global_orient: {global_orient.shape}")
        print(f"body_pose: {body_pose.shape}")
        print(f"transl: {transl.shape}")

        # Get SMPLX output
        output = self.smplx_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            return_verts=True,
        )

        # Get vertices and joints
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()
        
        print(f"\nVertices shape: {vertices.shape}")
        print(f"Joints shape: {joints.shape}")

        # Create camera and scene
        camera, camera_pose = self.create_camera()
        scene = self.create_scene(vertices, joints, camera, camera_pose)

        print(f"\nCamera pose:\n{camera_pose}")

        # Render
        color, depth = self.renderer.render(scene)
        print(f"\nRendered image shape: {color.shape}")
        print(f"Depth map shape: {depth.shape}")

        # Get projection matrix
        proj_matrix = self.get_projection_matrix()
        print(f"\nProjection matrix:\n{proj_matrix}")

        # Convert joints to camera space
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
        joints_2d[:, 0] = (joints_ndc[:, 0] + 1.0) * self.img_size * 0.5
        joints_2d[:, 1] = (1.0 - (joints_ndc[:, 1] + 1.0) * 0.5) * self.img_size
        
        print(f"\nFinal 2D joint positions (first 3):\n{joints_2d[:3]}")

        return color, joints_2d, depth

    def visualize(self, color, joints_2d, show_joints=True):
        """
        Visualize rendered mesh and projected joints
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(color)
        if show_joints:
            plt.scatter(joints_2d[:, 0], joints_2d[:, 1], c="r", s=20)
        plt.axis("off")
        plt.savefig("rendered_mesh.png")
        plt.close()


# Example usage
if __name__ == "__main__":
    # Initialize renderer
    model_path = "/home/paulj/projects/TADA/data/smplx/SMPLX_NEUTRAL_2020.npz"
    renderer = SMPLXRenderer(model_path)

    # Create some example pose
    betas = torch.zeros(1, 10)
    global_orient = torch.tensor([[0.5, 0.0, 0.0]])  # Slight rotation
    body_pose = torch.zeros(1, 63)

    # Example: Raise arms
    body_pose[0, 15:18] = torch.tensor([0, 0, 1.0])  # Left shoulder
    body_pose[0, 18:21] = torch.tensor([0, 0, -1.0])  # Right shoulder

    transl = torch.zeros(1, 3)

    # Render
    color, joints_2d, depth = renderer.render_mesh(
        betas=betas, global_orient=global_orient, body_pose=body_pose, transl=transl
    )

    # Visualize
    renderer.visualize(color, joints_2d)

    # Debug visualization of depth
    plt.figure(figsize=(10, 10))
    plt.imshow(depth)
    plt.colorbar()
    plt.title("Depth Map")