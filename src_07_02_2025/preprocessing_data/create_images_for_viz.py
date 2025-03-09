import os
import numpy as np
import pydicom
import tifffile
import pandas as pd


def open_dcm(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array
    img = img.astype(np.float32)
    
    if 'RescaleSlope' in dcm:
        rescale_slope = dcm.RescaleSlope
    else:
        rescale_slope = 1
            
    if 'RescaleIntercept' in dcm:
        rescale_intercept = dcm.RescaleIntercept
    else:
        rescale_intercept = 0
            
    img = img * rescale_slope + rescale_intercept
            
    return img

if __name__ == '__main__':
    
    folder_name = 'images_for_viz'
    os.makedirs(folder_name, exist_ok=True)

    tsv_path = '/project/outputs/tsv_files_train_test_valid_organized_table'
    
    train = pd.read_csv(os.path.join(tsv_path, 'train.tsv'), sep='\t')
    valid = pd.read_csv(os.path.join(tsv_path, 'validation.tsv'), sep='\t')
    test = pd.read_csv(os.path.join(tsv_path, 'test.tsv'), sep='\t')
    
    true_max = 0
    true_min = 2**17

    for i, row in train.iterrows():
        img_high = open_dcm(row['image_high'])
        img_low = open_dcm(row['image_low'])
        img_vmi = open_dcm(row['image_vmi_31'])
        img_lumem = open_dcm(row['image_lumen_36'])
        img_iondine = open_dcm(row['image_iodine_38'])

        true_max = max(true_max, img_high.max(), img_low.max(), img_vmi.max(), img_lumem.max(), img_iondine.max())
        true_min = min(true_min, img_high.min(), img_low.min(), img_vmi.min(), img_lumem.min(), img_iondine.min())

    print('True max:', true_max)
    print('True min:', true_min)

        