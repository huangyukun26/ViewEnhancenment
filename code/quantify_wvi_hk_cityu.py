import os
from natsort import ns, natsorted
from PIL import Image
from tqdm import tqdm
import numpy as np


def get_wvis_for_cityu():
    input_path="/lustre1/g/rec_fx/dp_paper/2024_WindowViewHK/output/"
    folders=os.listdir(input_path)

    for folder in folders:
            print(folder)

            id_count=0

        # for i in range(1):
            if "hk_cityu" in folder and "zip" not in folder and not os.path.exists("/lustre1/g/rec_fx/dp_paper/2024_WindowViewHK/output/test_log/hk_cityu_"+folder.split("_")[-1]+"_wvi_prop.csv"):

                with open("/lustre1/g/rec_fx/dp_paper/2024_WindowViewHK/output/test_log/hk_cityu_"+folder.split("_")[-1]+"_wvi_prop.csv", "w") as f:
                    f.write("no,name,building,green,sky,water,\n")
                    image_path= input_path+folder
                    files=os.listdir(image_path)

                    sorted_file_list=natsorted(files, alg=ns.PATH)

                    for file in tqdm(sorted_file_list):
                                label_file_name=input_path+"/"+folder+"/"+file
                                                        
                                im=Image.open(label_file_name)
                                        
                                im_numpy=np.array(im)

                                # color 
                                # 255, 255, 0   building
                                # 0  ,   0, 255 sky
                                # 0  , 255, 0   vegetation
                                # 96 ,  25, 134 waterbody
                                # 170, 170, 170 road

                                mask_building= np.all(im_numpy==[255,255,0],axis=-1)
                                label_building=np.sum(mask_building)/(900*900)

                                mask_vegetation=np.all(im_numpy==[0,255,0],axis=-1)
                                label_vegetation=np.sum(mask_vegetation)/(900*900)

                                mask_sky=np.all(im_numpy==[0, 0, 255],axis=-1)
                                label_sky=np.sum(mask_sky)/(900*900)

                                mask_water=np.all(im_numpy==[96, 25, 134],axis=-1)
                                label_water=np.sum(mask_water)/(900*900)

                                # mask_road=np.all(im_numpy==[170, 170, 170],axis=-1)
                                # label_road=np.sum(mask_road)/(900*900)

                                id_count+=1

                                f.write(str(id_count)+","+file+","+str(label_building)+","+str(label_vegetation)+","+str(label_sky)+","+str(label_water)+"\n")

if __name__=="__main__":
    get_wvis_for_cityu()