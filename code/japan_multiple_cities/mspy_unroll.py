import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# SOURCE: https://blogs.codingballad.com/unwrapping-the-view-transforming-360-panoramas-into-intuitive-videos-with-python-6009bd5bca94
import os
from PIL import Image
import numpy as np
from scipy.ndimage import map_coordinates
from tqdm import tqdm


def map_to_sphere(x, y, z, W, H, f, yaw_radian, pitch_radian):


    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    # Apply rotation transformations here
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))

    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                           np.cos(theta) * np.sin(pitch_radian),
                           np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()


def interpolate_color(coords, img, method='bilinear'):
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama_path, FOV, output_size, yaw, pitch):
    panorama = Image.open(panorama_path).convert('RGB')
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)

    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    x = u - W / 2
    y = H / 2 - v
    z = f

    theta, phi = map_to_sphere(x, y, z, W, H, f, yaw_radian, pitch_radian)

    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))

    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')
    # output_image.show()


    return output_image



# Input folder
folder_path = '/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/uzip/'

# Output folder
output_folder = '/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/unrolled/'

# Confirm if the output folder exists, create if not
os.makedirs(output_folder, exist_ok=True)

image_list=['kyoto_1', 'kyoto_2', 'sapporo_b3', 'sapporo_1', 'sapporo_2', 'sapporo_3', 'sapporopano_b2']

for path in image_list:
# Process every image
    for filename in tqdm(os.listdir(folder_path+path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Check file extension
            image_path = os.path.join(folder_path+path, filename)
            image = Image.open(image_path)

            # Get width and height of images
            width, height = image.size

            # Calculate new width and height
            w, h = np.around(3*width/4).astype(int), np.around(3*height/4).astype(int)

            # Process and save images
            for deg in [90,270]:  # Updated to include 0, 90, 180, 270 degrees
                output_filename = f'output_{os.path.splitext(filename)[0]}_{deg}.jpg'
                output_path = os.path.join(output_folder, output_filename)

                # Check if the output file already exists
                if not os.path.exists(output_path):
                    output_image = panorama_to_plane(image_path, 120, (w, h), deg, 90)
                    output_image.save(output_path)
                else:
                    print(f"File {output_filename} already exists. Skipping...")


