
# SOURCE: https://blogs.codingballad.com/unwrapping-the-view-transforming-360-panoramas-into-intuitive-videos-with-python-6009bd5bca94
import os
from PIL import Image
import numpy as np
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import torch

def interpolate_color(coords, img, method='bilinear'):
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def map_to_sphere_torch(x, y, z, yaw_radian, pitch_radian):
    norm = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = torch.acos(z / norm)
    phi = torch.atan2(y, x)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_pitch = torch.sin(pitch_radian)
    cos_pitch = torch.cos(pitch_radian)

    theta_prime = torch.acos(sin_theta * sin_phi * sin_pitch + cos_theta * cos_pitch)
    phi_prime = torch.atan2(
        sin_theta * sin_phi * cos_pitch - cos_theta * sin_pitch,
        sin_theta * cos_phi
    )
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime, phi_prime


def panorama_to_plane(panorama_image, FOV, output_size, yaw, pitch, device='cuda'):
    panorama = panorama_image.convert('RGB')
    pano_array = np.array(panorama).astype(np.float32) / 255.0
    pano_tensor = torch.from_numpy(pano_array).permute(2, 0, 1).unsqueeze(0).to(device)

    pano_height, pano_width = pano_array.shape[:2]

    yaw_radian = torch.tensor(np.radians(yaw), dtype=torch.float32, device=device)
    pitch_radian = torch.tensor(np.radians(pitch), dtype=torch.float32, device=device)

    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)

    # 网格坐标
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    u = u.to(device).float()
    v = v.to(device).float()

    x = u - W / 2
    y = H / 2 - v
    z = torch.full_like(x, f)

    theta, phi = map_to_sphere_torch(x, y, z, yaw_radian, pitch_radian)

    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    grid_U = (U / (pano_width - 1)) * 2 - 1
    grid_V = (V / (pano_height - 1)) * 2 - 1
    grid = torch.stack([grid_U, grid_V], dim=-1).unsqueeze(0)

    output = torch.nn.functional.grid_sample(pano_tensor, grid, mode='bilinear', align_corners=True)

    output_image = (output.squeeze().permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(output_image)



# 设置 CUDA / CPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 文件夹路径
# Input folder
folder_path = '/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/uzip/'

# Output folder
output_folder = '/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/unrolled/'

os.makedirs(output_folder, exist_ok=True)

image_list=['kyoto_1', 'kyoto_2', 'sapporo_b3', 'sapporo_1', 'sapporo_2', 'sapporo_3', 'sapporopano_b2']

# 获取图像列表（只包含 JPG 和 PNG）
image_files=[]
for path in image_list:
    image_files.extend([folder_path+path+"/"+f for f in os.listdir(folder_path+path) if f.endswith(('.jpg', '.png'))])

# 处理函数（单张图像）
def process_image(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    base_path=os.path.dirname(filename)

    # 如果两个目标文件都存在，直接跳过
    if all(os.path.exists(os.path.join(output_folder, f'output_{base}_{deg}.jpg')) for deg in [90, 270]):
        return

    try:
        image_path = filename
        with Image.open(image_path) as img:
            img.load()
            image = img.copy()

        width, height = image.size
        w, h = np.around(3 * width / 4).astype(int), np.around(3 * height / 4).astype(int)

        for deg in [90, 270]:
            output_filename = f'output_{base}_{deg}.jpg'
            output_path = os.path.join(output_folder, output_filename)

            if not os.path.exists(output_path):
                output_image = panorama_to_plane(image, 120, (w, h), deg, 90, device=device)
                output_image.save(output_path)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# 使用多线程并行处理
max_workers = min(16, os.cpu_count())  # 自适应线程数，最多 8 个
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(process_image, image_files), total=len(image_files), desc="Processing"))
