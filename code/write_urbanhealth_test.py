
import os
from tqdm import tqdm

input_path="/lustre1/g/rec_fx/data_processing/sam_windowview/ImageSets/ImageSetsAll/hk_extra_10_tpu/"

image_path="/lustre1/g/rec_fx/model_train_log/deeplabv3_new/verification/20240426_greenery_exposure/input/"

# folders=os.scandir(image_path)

# count=0

# with open(input_path+"test.txt", "w") as f:

#     for folder in folders:

#         if ".zip" not in folder.name:
#             # if "batch_8" in folder:
#                 print(folder.name)
#                 files=os.scandir(image_path+folder.name)
#                 for file in tqdm(files):
#                     if ".png" in file.name:
#                         count+=1
#                         f.write(folder.name+"/"+file.name+"\n")

# print(count)



folders=os.listdir(image_path)

count=0


with open(input_path+"test.txt", "w") as f:
            for folder in folders:
                if ".zip" not in folder:
                # if "batch_9" in folder:
                    print(folder)
                    files=os.listdir(image_path+folder)
                    for file in tqdm(files):
                        if ".png" in file:
                            count+=1
                            f.write(folder+"/"+file+"\n")

print(count)