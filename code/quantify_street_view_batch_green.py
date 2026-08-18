import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/lustre1/g/rec_fx/dp_paper/2024_3DGreeneryExposure')
    parser.add_argument('--segmented_streetview_folder', type=str, default='output_dingtian')
    parser.add_argument('--output_path', type=str, default='/lustre1/g/rec_fx/dp_paper/2024_3DGreeneryExposure/sv_quantification/')
    parser.add_argument('--csv_name', type=str, default='/lustre1/g/rec_fx/dp_paper/2024_3DGreeneryExposure/')
    args = parser.parse_args()

    # change the input path as the base folder
    input_path = args.input_path
    segmented_streetview_folder=args.segmented_streetview_folder
    output_path=args.output_path
    csv_name=args.csv_name



    target_colors = {
        (4, 200, 3): 'tree',
        (4, 250, 7): 'grass',
        (204, 255, 4): 'plant_life',
        (255, 0, 0): 'flower',
        (0, 82, 255): 'palm'
    }

    columns = ["pic"] + list(target_colors.values())
    df = pd.DataFrame(columns=columns)

    def process_images_in_directory(directory, processed_names):
            for filename in processed_names:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    pic_name = filename.replace(".png_mask.png","")
                    print(pic_name)

                    with Image.open(os.path.join(directory, filename)) as img:
                        img_array = np.array(img)[:, :, :3] 
                        total_pixels = img_array.shape[0] * img_array.shape[1]

                        color_counts = {color: 0 for color in target_colors}
                        for color in target_colors:
                            color_counts[color] = np.sum(np.all(img_array == color, axis=-1))

                        proportions = {label: color_counts[color] / total_pixels 
                                    for color, label in target_colors.items()}

                        df.loc[len(df)] = [pic_name] + list(proportions.values())
                        # break


    subdir_path = os.path.join(input_path, segmented_streetview_folder)
    processed_names=os.listdir(subdir_path)
    process_images_in_directory(subdir_path, processed_names)

    df.fillna(0, inplace=True)

    df.to_csv(output_path+csv_name, index=False)