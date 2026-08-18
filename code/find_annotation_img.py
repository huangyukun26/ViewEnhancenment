import os
import pandas as pd
import shutil
from tqdm import tqdm

input_csv="/home/maosuli/2dsegmentation/sam_landfill/japan_120_name.csv"
image_path="/lustre1/g/rec_fx/dp_paper/2024_3DGreeneryExposure/batch_1/"
output_path="/lustre1/g/rec_fx/dp_paper/2024_3DGreeneryExposure/annotation_120/"

df=pd.read_csv(input_csv)

for idx,row in tqdm(df.iterrows()):
    shutil.copy(image_path+row['name'],output_path+row['name'])

