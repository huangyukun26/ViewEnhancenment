import os

input_path="/lustre1/g/rec_fx/data_processing/sam_windowview/ImageSets/ImageSetsAll/all_gee_1102/w_japan_as_train/val.txt"

with open(input_path,"r") as f:
    lines=f.readlines()
    for line in lines:
        if os.path.exists("/lustre1/g/rec_fx/data_processing/sam_windowview/ann/ann_four_type_all_gee/"+line.strip()):
            print(line)

        if os.path.exists("/lustre1/g/rec_fx/data_processing/sam_windowview/img/images_all_gee_1102/"+line.strip()):
            print(line)            