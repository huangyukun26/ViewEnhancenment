import os
import pandas as pd



def count_file_no():
    path1="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/SapporoPanoRaw/sapporopano_b2"
    path2="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/SapporoPanoRaw/sapporo_b3"
    path3="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/uzip/sapporo_1/"
    path4="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/uzip/sapporo_2/"
    path5="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/uzip/sapporo_3/"

    path_list=[path1, path2, path3, path4, path5]

    file_list=[]

    count=0
    for path in path_list:
        count+=len(os.listdir(path))

        for file in os.listdir(path):
            file_list.append(file)

    print(count)

    no_duplicate=list(set(file_list))
    print(len(no_duplicate))


import shutil
from tqdm import tqdm
def remove_park_files():
    input_csv="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/SapporoPanoRaw/pano_within_park_sapporo_2757.csv"
    df=pd.read_csv(input_csv)

    output_path="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/SapporoPanoRaw/withinpark/"

    path1="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/SapporoPanoRaw/sapporopano_b2/"
    path2="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/SapporoPanoRaw/sapporo_b3/"
    path3="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/uzip/sapporo_1/"
    path4="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/uzip/sapporo_2/"
    path5="/lustre1/g/rec_fx/dp_paper/2025_JapanWellBeingMultipleCities/StreetView/uzip/sapporo_3/"

    path_list=[path1, path2, path3, path4, path5]

    for path in tqdm(path_list):

        for file in os.listdir(path):
            if file in df['filename'].values.tolist():
                shutil.move(path+file,output_path+file)

                # break

if __name__=="__main__":
    remove_park_files()

