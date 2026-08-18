import pandas as pd
import os
import shutil
from tqdm import tqdm

input_path="/home/maosuli/dp_paperdata/20251007_dongwei_distance/left.csv"

segment_path="/lustre1/g/rec_fx/dp_paper/2024_WindowViewHK/output/"

output_path="/home/maosuli/dp_paperdata/20251007_dongwei_distance/output/"

df=pd.read_csv(input_path)

path_list=[]

folders=os.listdir(segment_path)

for idx, row in tqdm(df.iterrows()):
    for folder in folders:
        if ".zip" not in folder and "hk" in folder:
            # from IPython import embed
            # embed()

            if os.path.exists(segment_path+folder+"/"+row['name_x'][:-9]+"_pred.png"):
                shutil.copy(segment_path+folder+"/"+row['name_x'][:-9]+"_pred.png",output_path+row['name_x'][:-9]+"_pred.png")
                # print("yes")

        