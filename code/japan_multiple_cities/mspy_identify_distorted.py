import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# 配置路径
test_image_folder = "/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/uzip/" #all street view image
output_csv_path = "/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/csv/test_predictions.csv"
corrupted_csv_path = "/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/csv/corrupted_images.csv"
model_save_path = "/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/model/best_model.pth"

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available() )

# 加载最佳模型
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # 遍历整个文件夹，筛选未损坏的图片
# def find_valid_images(root_folder, extension=".jpg"):
#     valid_images = []
#     corrupted_images = []

#     all_images = []
#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in sorted(filenames):
#             if filename.endswith(extension):
#                 if "kyoto_" in dirpath or "sapporo_" in dirpath or "sapporopano_b2" in dirpath:
#                     image_path = os.path.join(dirpath, filename)
#                     all_images.append(image_path)

#     print(f"共检测到 {len(all_images)} 张图片，开始检查图片完整性...")

#     for image_path in tqdm(all_images, desc="检查图片完整性"):
#         try:
#             img = Image.open(image_path)
#             img.verify()  # 检查图片是否损坏
#             valid_images.append(image_path)
#         except Exception as e:
#             print(f"无法识别的图片 (跳过): {image_path} - {e}")
#             corrupted_images.append(image_path)

#     return valid_images, corrupted_images

# # 执行筛选
# test_images, corrupted_images = find_valid_images(test_image_folder)
# print(f"共找到 {len(test_images)} 张未损坏的图片，开始预测...")
# print(f"共有 {len(corrupted_images)} 张图片损坏，将保存损坏列表")


all_images = []
for dirpath, _, filenames in os.walk(test_image_folder):
        for filename in sorted(filenames):
            if filename.endswith(".jpg"):
                if "kyoto_" in dirpath or "sapporo_" in dirpath or "sapporopano_b2" in dirpath:
                    image_path = os.path.join(dirpath, filename)
                    all_images.append(image_path)

# ✅ 预测函数
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"读取失败，跳过: {image_path} - {e}")
        return None

    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted_label = torch.argmax(output).item()
    return predicted_label  # 0=错误, 1=正常

# 批量预测（带 tqdm 进度条）
test_predictions = []
for image_path in tqdm(all_images, desc="图片预测中"):
    image_name = os.path.basename(image_path)
    label = predict_image(image_path)
    if label is not None:
        test_predictions.append([image_name, label])

# 保存预测结果
df_test_predictions = pd.DataFrame(test_predictions, columns=["image_name", "error_label"])
df_test_predictions.to_csv(output_csv_path, index=False)
print(f" 预测完成，结果已保存至 {output_csv_path}")

# # 保存损坏图片列表
# df_corrupted = pd.DataFrame(corrupted_images, columns=["corrupted_image_path"])
# df_corrupted.to_csv(corrupted_csv_path, index=False)
# print(f"损坏图片列表已保存至 {corrupted_csv_path}")
