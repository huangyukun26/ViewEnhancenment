
import os
from tqdm import tqdm

input_path="/lustre1/g/rec_fx/data_processing/sam_windowview/ImageSets/ImageSetsAll/landsD_finetune/"

image_path="/lustre1/g/rec_fx/dp_paper/20260121_RestHKLandsDModels/"

folders=os.scandir(image_path)

count=0


for folder in folders:
    if ".zip" not in folder.name:
        with open(input_path+"test_"+folder.name+".txt", "w") as f:

                    # if "batch_8" in folder:
                        print(folder.name)
                        files=os.scandir(image_path+folder.name)
                        for file in tqdm(files):
                            if ".png" in file.name:
                                count+=1
                                f.write(folder.name+"/"+file.name+"\n")

                        # break

print(count)