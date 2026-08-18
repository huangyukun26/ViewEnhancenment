
import os
from tqdm import tqdm

input_path="/lustre1/g/rec_fx/data_processing/sam_windowview/ImageSets/ImageSetsAll/japan_only/"

image_path="/lustre1/g/rec_fx/dp_paper/2024_3DGreeneryExposure/new_bldg_2w_1/"

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



files=os.listdir(image_path)

count=0

# for file in tqdm(files):
    # if ".zip" not in file:
with open(input_path+"japan_test_1.txt", "w") as f:
                # if "batch_9" in folder:
                    # print(folder)
                    # files=os.listdir(image_path+folder)
                    for file in tqdm(files):
                        if ".png" in file:
                            count+=1
                            f.write(file+"\n")

print(count)