import torch

import hydra
from lib.dlmesh import DLMesh
import trimesh,pyrender


import numpy as np
import trimesh
import pyrender
from PIL import Image

# Assuming `model` and `pyrender` are already imported and initialized


def render_mesh_to_image(mesh, width=640, height=480, camera_distance=2.0):
    scene = pyrender.Scene()

    # Calculate the center of the mesh
    center = mesh.bounds.mean(axis=0)

    # Define the camera's position to the side of the mesh
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = center + [
        camera_distance,
        0,
        0,
    ]  # Position camera to the side (x-axis)

    # Rotate the camera to look at the center of the mesh
    direction = center - camera_pose[:3, 3]
    direction /= np.linalg.norm(direction)
    up = np.array([0, 0, 1])  # Assuming Z-up coordinate system
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    camera_pose[:3, :3] = np.vstack([right, up, -direction]).T

    # Create a camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)

    # Add a directional light for better visualization
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    # Add the mesh to the scene
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    # Render the scene off-screen
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, _ = renderer.render(scene)
    renderer.delete()

    return Image.fromarray(color)


@hydra.main(config_path="configs", config_name="tada_wo_dpt.yaml")
def main(cfg):
    model = DLMesh(cfg.model)
    chpt = torch.load("/home/paulj/Downloads/dg_native_resol_17_ep0100.pth")
    model_raw_albedo = model.raw_albedo
    model.load_state_dict(chpt['model'])


    # Parameters
    image_width = 640
    image_height = 480
    num_frames = 16

    # List to hold rendered images
    images = []

    for i in range(num_frames):
        mesh_out = model.get_mesh(False, i)
        mesh_out = mesh_out[0]
        vertices, faces = (
            mesh_out.v.detach().cpu().numpy(),
            mesh_out.f.detach().cpu().numpy(),
        )
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Render the mesh to an image
        image = render_mesh_to_image(trimesh_mesh, width=image_width, height=image_height)
        images.append(image)

    # Combine images into a strip
    strip_width = image_width * num_frames
    strip_image = Image.new("RGB", (strip_width, image_height))

    for idx, img in enumerate(images):
        strip_image.paste(img, (idx * image_width, 0))

    # Save the strip image
    strip_image.save("motion_strip.png")

    # If you need to display the strip image
    strip_image.show()


if __name__ == "__main__":
    main()
