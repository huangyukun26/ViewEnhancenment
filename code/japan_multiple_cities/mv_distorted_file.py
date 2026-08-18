import os
import shutil
import pandas as pd
from tqdm import tqdm

# **路径配置**
csv_path = "/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/csv/test_predictions.csv"       # CSV 文件路径
image_folder = "/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/uzip/"                          # 原始图片存放路径
error_images_folder = "/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/error_image/"    # 目标文件夹 (存放 error_label = 0 的图片)

# **创建目标文件夹**
os.makedirs(error_images_folder, exist_ok=True)

# **读取 CSV**
df = pd.read_csv(csv_path)

# **筛选 error_label = 0 的图片**
error_images = df[df['error_label'] == 0]
print(f"🔎 共有 {len(error_images)} 张 error_label=0 的图片需要移动")

image_list=['kyoto_1', 'kyoto_2', 'sapporo_b3', 'sapporo_1', 'sapporo_2', 'sapporo_3', 'sapporopano_b2']

# **遍历并移动图片（加 tqdm 进度条）**
for _, row in tqdm(error_images.iterrows(), total=len(error_images), desc="移动图片中"):
    image_name = row["image_name"]
    for path in image_list:
        if os.path.exists(image_folder+path+"/"+image_name):
            source_path = os.path.join(image_folder+path+"/", image_name)
            target_path = os.path.join(error_images_folder, image_name)

    # 确保文件存在再移动
    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
        tqdm.write(f"✅ 已移动: {image_name}")
    else:
        tqdm.write(f"⚠️ 文件未找到: {source_path}")

print(f"🎉 移动完成！错误图片已存放于 {error_images_folder}")
