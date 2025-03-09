import os
import numpy as np
import pydicom
import pandas as pd
import tifffile
import torch

def open_dcm(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)

    rescale_slope = float(dcm.RescaleSlope) if 'RescaleSlope' in dcm else 1
    rescale_intercept = float(dcm.RescaleIntercept) if 'RescaleIntercept' in dcm else 0

    return img * rescale_slope + rescale_intercept

if __name__ == '__main__':
    tsv_path = '/project/outputs/tsv_files_train_test_valid_organized_table'

    train = pd.read_csv(os.path.join(tsv_path, 'train.tsv'), sep='\t')
    val = pd.read_csv(os.path.join(tsv_path, 'validation.tsv'), sep='\t')


    true_max = float('-inf')
    true_min = float('inf')
    images_list = []

    for _, row in train.iterrows():
        img_high = open_dcm(row['image_high'])
        img_low = open_dcm(row['image_low'])
        img_vmi = open_dcm(row['image_vmi_31'])
        img_lumen = open_dcm(row['image_lumen_36'])

        true_max = max(true_max, img_high.max(), img_low.max(), img_vmi.max(), img_lumen.max())
        true_min = min(true_min, img_high.min(), img_low.min(), img_vmi.min(), img_lumen.min())

        images_list.extend([img_high, img_low, img_vmi, img_lumen])  

    vec_all_images = np.concatenate([img.flatten() for img in images_list], axis=0)

    print(vec_all_images.shape)
    print(f'Mean: {vec_all_images.mean()}')
    print(f'Std: {vec_all_images.std()}')
    print(f'True max: {true_max}')
    print(f'True min: {true_min}')

    exit()
    mean = vec_all_images.mean()
    std = vec_all_images.std()
    path_out = '/project/outputs/images_for_viz'
    os.makedirs(path_out, exist_ok=True)

    for _, row in val.iterrows():
        if _ == 5:
            break

        img_name = row['image_high'].split('/')[-1].split('.')[0]

        img_high = open_dcm(row['image_high'])
        img_low = open_dcm(row['image_low'])
        img_vmi = open_dcm(row['image_vmi_31'])
        img_lumen = open_dcm(row['image_lumen_36'])

        img_high_norm1 = (img_high - true_min) / (true_max - true_min)
        img_low_norm1 = (img_low - true_min) / (true_max - true_min)
        img_vmi_norm1 = (img_vmi - true_min) / (true_max - true_min)

        img_high_norm2 = (img_high - mean) / std
        img_low_norm2 = (img_low - mean) / std
        img_vmi_norm2 = (img_vmi - mean) / std

        tifffile.imwrite(os.path.join(path_out, f'{img_name}_high.tif'), img_high)
        tifffile.imwrite(os.path.join(path_out, f'{img_name}_low.tif'), img_low)
        tifffile.imwrite(os.path.join(path_out, f'{img_name}_vmi.tif'), img_vmi)

        tifffile.imwrite(os.path.join(path_out, f'{img_name}_high_min_max.tif'), img_high_norm1)
        tifffile.imwrite(os.path.join(path_out, f'{img_name}_low_min_max.tif'), img_low_norm1)
        tifffile.imwrite(os.path.join(path_out, f'{img_name}_vmi_min_max.tif'), img_vmi_norm1)

        tifffile.imwrite(os.path.join(path_out, f'{img_name}_high_mean.tif'), img_high_norm2)
        tifffile.imwrite(os.path.join(path_out, f'{img_name}_low_mean.tif'), img_low_norm2)
        tifffile.imwrite(os.path.join(path_out, f'{img_name}_vmi_mean.tif'), img_vmi_norm2)

        img_high = torch.from_numpy(img_high).unsqueeze(0).unsqueeze(0)
        img_low = torch.from_numpy(img_low).unsqueeze(0).unsqueeze(0)
        img_vmi = torch.from_numpy(img_vmi).unsqueeze(0).unsqueeze(0)
        img_lumen = torch.from_numpy(img_lumen).unsqueeze(0).unsqueeze(0)
        
        img_concat = torch.cat([img_high, img_vmi, img_lumen], dim=1).numpy()
        print(img_concat.shape)
        tifffile.imwrite(os.path.join(path_out, f'{img_name}_concat.tif'), img_concat)

        img_concat_norm1 = (img_concat - true_min) / (true_max - true_min)
        img_concat_norm2 = (img_concat - mean) / std

        tifffile.imwrite(os.path.join(path_out, f'{img_name}_concat_min_max.tif'), img_concat_norm1)
        tifffile.imwrite(os.path.join(path_out, f'{img_name}_concat_mean.tif'), img_concat_norm2)