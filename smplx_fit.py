import torch
import numpy as np
import smplx
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import trimesh

from lib.common.renderer import Renderer
from lib.common.utils import safe_normalize

# COCO Keypoint mapping
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# SMPLX to COCO joint mapping
SMPLX_TO_COCO = {
    'nose': 55,
    'left_eye': 57,
    'right_eye': 56,
    'left_ear': 59,
    'right_ear': 58,
    'left_shoulder': 17,
    'right_shoulder': 16,
    'left_elbow': 19,
    'right_elbow': 18,
    'left_wrist': 21,
    'right_wrist': 20,
    'left_hip': 2,
    'right_hip': 1,
    'left_knee': 5,
    'right_knee': 4,
    'left_ankle': 8,
    'right_ankle': 7,
}

# Create ordered list of SMPLX indices matching COCO order
SMPLX_JOINT_IDS = [SMPLX_TO_COCO[name] for name in COCO_KEYPOINTS]

# Load Detectron2 Keypoint R-CNN model (or use HRNet as shown previously)
def load_keypoint_rcnn():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    )
    return DefaultPredictor(cfg)


# Extract 2D keypoints
def extract_keypoints(image_path, predictor):
    image = cv2.imread(image_path)
    outputs = predictor(image)
    keypoints = outputs["instances"].pred_keypoints[0].cpu().numpy()
    return keypoints  


def visualize_keypoints(image_path, keypoints, output_path=None):
    # Load image
    image = cv2.imread(image_path)
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


# Fit SMPL model using SMPLify-like fitting
def fit_smpl_to_keypoints(smpl_model,keypoints, model_path, device="cuda"):
    # Initialize the SMPL model


    # Initial pose, shape, and camera parameters
    pose_params = torch.zeros(1, 66, device=device).detach().requires_grad_(True)
    shape_params = torch.zeros(1, 10, device=device).detach().requires_grad_(True)
    camera_params = torch.tensor(
        [1.0, 0.0, 0.0], device=device
    ).unsqueeze(0).detach().requires_grad_(True)  # [scale, tx, ty]

    # Optimization
    optimizer = torch.optim.Adam([pose_params, shape_params, camera_params], lr=0.01)

    # Target keypoints in COCO format [1, 17, 3] (x, y, confidence)
    keypoints_2d = torch.tensor(keypoints, device=device).unsqueeze(0)
    
    # Get confidence weights from keypoints
    confidence_weights = keypoints_2d[..., 2]
    keypoints_2d = keypoints_2d[..., :2]  # Only keep x,y coordinates

    for step in range(10):
        optimizer.zero_grad()

        # Forward pass through SMPL
        smpl_output = smpl_model(
            betas=shape_params,
            body_pose=pose_params[:, 3:],
            global_orient=pose_params[:, :3],
            transl=camera_params,
        )

        # Select corresponding SMPLX joints
        joints_3d = smpl_output.joints[:, SMPLX_JOINT_IDS, :]

        # Project 3D joints to 2D using weak perspective camera model
        projected_joints = camera_params[:, 0].unsqueeze(-1).unsqueeze(-1) * joints_3d[..., :2]
        projected_joints = projected_joints + camera_params[:, 1:].unsqueeze(1)

        # Weighted MSE loss between detected and projected 2D keypoints
        loss = torch.sum(confidence_weights.unsqueeze(-1) * 
                        (projected_joints - keypoints_2d) ** 2)
        
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f'Step {step}: Loss = {loss.item():.4f}')

    return smpl_output, projected_joints


def render_mesh(vertices, faces, image_size=512):
    # Convert vertices and faces to Mesh object
    mesh = Mesh(vertices, faces)
    
    # Initialize renderer
    renderer = Renderer()
    
    # Create simple camera MVP matrix (perspective projection)
    fovy = np.deg2rad(60)
    aspect = 1.0
    near = 0.1
    far = 10
    
    proj = torch.from_numpy(
        perspective(fovy, aspect, near, far)
    ).cuda().float()
    
    # Position camera looking at origin
    cam_pos = torch.tensor([0, 0, -2.0], device='cuda')
    look_at = torch.tensor([0, 0, 0], device='cuda')
    up = torch.tensor([0, 1, 0], device='cuda')
    
    view = torch.from_numpy(
        lookat(cam_pos.cpu().numpy(), 
               look_at.cpu().numpy(),
               up.cpu().numpy())
    ).cuda().float()
    
    mvp = torch.matmul(proj, view)[None,...]
    
    # Create simple directional light
    light_d = safe_normalize(torch.tensor([1.0, 1.0, -1.0], device='cuda'))
    
    # Create simple vertex colors
    vertex_colors = torch.ones_like(vertices, device='cuda')
    
    # Render
    alpha, color = renderer(
        mesh,
        mvp,
        light_d=light_d
    )
    
    # Convert to image
    image = color[0].cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return image

def perspective(fovy, aspect, near, far):
    """Returns perspective projection matrix"""
    f = 1.0 / np.tan(fovy / 2)
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0], 
        [0, 0, (far+near)/(near-far), 2*far*near/(near-far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def lookat(eye, at, up):
    """Returns lookat view matrix"""
    z = eye - at
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    
    return np.array([
        [x[0], x[1], x[2], -np.dot(x, eye)],
        [y[0], y[1], y[2], -np.dot(y, eye)],
        [z[0], z[1], z[2], -np.dot(z, eye)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

# Main
if __name__ == "__main__":
    image_path = "/home/paulj/projects/TADA/rgb.png"  # Change this to your image path
    model_path = "/home/paulj/projects/TADA/data/smplx/SMPLX_NEUTRAL_2020.npz"  # Path to the folder with SMPL model files
    smpl_model = smplx.create(
        model_path, model_type="smpl", gender="neutral", batch_size=1
    ).to("cuda")
    # Load keypoint detector and extract keypoints
    predictor = load_keypoint_rcnn()
    keypoints = extract_keypoints(image_path, predictor)
    visualize_keypoints(image_path, keypoints,output_path="keypoints.png")
    # Fit SMPL model to keypoints
    exit()
    smpl_output, projected_joints = fit_smpl_to_keypoints(
        smpl_model, keypoints, model_path
    )

    # The output `smpl_output` contains 3D vertices, joints, etc., for visualization or further processing.

    # Visualize the fitted SMPL model using pyrender

    # Visualize the fitted SMPL model using pyrender
    import pyrender
    from pyrender import Mesh, Node, Scene, Viewer
    from pyrender.constants import RenderFlags
    from pyrender.light import PointLight

    # Get vertices from SMPL output
    vertices = smpl_output.vertices[0].detach()
    faces = smpl_model.faces

    # Render using nvdiffrast
    # rendered_image = render_mesh(vertices, faces)
    # cv2.imwrite("rendered_smpl.png", cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))

    # Also visualize with pyrender for comparison
    mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices.cpu().numpy(), faces)
    )

    scene = pyrender.Scene()
    scene.add(mesh)

    pyrender.Viewer(scene, use_raymond_lighting=True)
