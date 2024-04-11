import torch
from matplotlib import pyplot as plt

landmarks = torch.load("test.pt")



# Function to create the image tensor with points
def create_image_with_points(points, image_size):
    image = torch.zeros(image_size)
    for point in points:
        x, y = point * image_size
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            image[int(x), int(y)] = 1
    return image


# Example points (replace this with your actual points)
points = landmarks.detach().cpu().numpy()[0,:]
# Image size (width, height)
image_size = (256, 256)

# Create the image tensor
image = create_image_with_points(points, image_size)

# Convert tensor to NumPy array
image_np = image.numpy()

# Display the image
plt.imshow(image_np, cmap="gray")
plt.axis("off")
plt.show()

# Save the image
plt.imsave("points_image.png", image_np, cmap="gray")
