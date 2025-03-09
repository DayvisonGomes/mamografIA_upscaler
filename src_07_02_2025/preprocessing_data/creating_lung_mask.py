import os
import pandas as pd
import pydicom
import numpy as np
import tifffile

def window_ct(dcm, w, c, ymin, ymax):
    """Windows a CT slice.

    Args:
        dcm (pydicom.dataset.FileDataset): The DICOM dataset containing the CT slice.
        w: Window Width parameter.
        c: Window Center parameter.
        ymin: Minimum output value.
        ymax: Maximum output value.

    Returns:
        Windowed slice.
    """
    b = dcm.RescaleIntercept
    m = dcm.RescaleSlope
    x = m * dcm.pixel_array + b

    y = np.zeros_like(x)
    y[x <= (c - 0.5 - (w - 1) / 2)] = ymin
    y[x > (c - 0.5 + (w - 1) / 2)] = ymax
    mask = (x > (c - 0.5 - (w - 1) / 2)) & (x <= (c - 0.5 + (w - 1) / 2))
    y[mask] = ((x[mask] - (c - 0.5)) / (w - 1) + 0.5) * (ymax - ymin) + ymin

    return y

def normalize_image(image, ymin, ymax):
    return (image - ymin) / (ymax - ymin)

def create_lung_mask(image, threshold=0.5):
    """Creates a binary mask to highlight lung structures.

    Args:
        image (np.array): The windowed CT image.
        threshold (float): Relative threshold for segmentation.

    Returns:
        np.array: Binary mask with lung structures highlighted.
    """
    mask = np.zeros_like(image)
    mask[image > threshold] = 1.0
    return mask

if __name__ == '__main__':
    base_path = r'D:\Users\dayv\mamografIA_upscaler\outputs\tsv_files_train_valid_pulmao'
    output_path = r'D:\Users\dayv\mamografIA_upscaler\data\data_lung_masks'
    os.makedirs(output_path, exist_ok=True)
    
    df_train = pd.read_csv(os.path.join(base_path, 'train.tsv'), sep='\t')
    df_test = pd.read_csv(os.path.join(base_path, 'test.tsv'), sep='\t')
    df_val = pd.read_csv(os.path.join(base_path, 'validation.tsv'), sep='\t')
    
    df_list = [df_train, df_test, df_val]
    
    for df in df_list:
        
        for i, row in df.iterrows():
            path = row['image']
            path = path.replace('/project/ct_noel', 'D:/Users/bbruno/Data/ct_noel')
            
            name = path.split('/')[-1]
            
            dcm = pydicom.dcmread(path)
            window_center = float(dcm.WindowCenter)
            window_width = float(dcm.WindowWidth)
            ymin = 0
            ymax = 1

            windowed_image = window_ct(dcm, window_width, window_center, ymin, ymax)
            normalized_image = normalize_image(windowed_image, ymin, ymax)

            lung_mask = create_lung_mask(normalized_image, threshold=0.5)
            
            output_file_path = f'{name}_mask.tiff'
            final_path = os.path.join(output_path, output_file_path)
            tifffile.imwrite(final_path, lung_mask.astype(np.float32))
            
            del windowed_image
            del normalized_image
            del lung_mask
            
        print('1')

            
            
            