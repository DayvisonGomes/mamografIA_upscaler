import os
import pydicom
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import argparse

def crop_images(base_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    max_top = 0
    max_left = 0
    max_bottom = float('inf')
    max_right = float('inf')

    # for root, dirs, files in os.walk(folder_path):
    #     for file in files:
    #         if file.endswith('.dcm'):
    #             file_path = os.path.join(root, file)
    #             dicom_data = pydicom.dcmread(file_path)
    #             pixel_array = dicom_data.pixel_array

    folders = os.listdir(base_path)
    for folder in folders:
        if '-recon' in folder:
            files = os.listdir(os.path.join(base_path, folder))
            file_path = os.path.join(base_path, folder, files[0])
            dicom_data = pydicom.dcmread(file_path)
            pixel_array = dicom_data.pixel_array
            
            top, left, bottom, right = find_non_black_region(pixel_array)

            max_top = max(max_top, top)
            max_left = max(max_left, left)
            max_bottom = min(max_bottom, bottom)
            max_right = min(max_right, right)

    crop_height = max_bottom - max_top
    crop_width = max_right - max_left

    num_crops_h = (crop_height - 512) // 512 + 1
    num_crops_w = (crop_width - 512) // 512 + 1

    # for root, dirs, files in os.walk(base_path):
    #     for file in files:
    #         if file.endswith('.dcm'):
    #             file_path = os.path.join(root, file)
    #             dicom_data = pydicom.dcmread(file_path)
    #             pixel_array = dicom_data.pixel_array
                
    folders = os.listdir(base_path)
    for folder in folders:
        if '-recon' in folder:
            files = os.listdir(os.path.join(base_path, folder))
            file_path = os.path.join(base_path, folder, files[0])
            dicom_data = pydicom.dcmread(file_path)
            pixel_array = dicom_data.pixel_array
            
            cropped_image = pixel_array[max_top:max_bottom, max_left:max_right]
            additional_crop_top = 100
            additional_crop_bottom = cropped_image.shape[0] - 100
            additional_crop_left = 0
            additional_crop_right = 4000

            cropped_image = cropped_image[additional_crop_top:additional_crop_bottom, additional_crop_left:additional_crop_right]
            num_crops_h = (cropped_image.shape[0] - 512) // 512 + 1
            num_crops_w = (cropped_image.shape[1] - 512) // 512 + 1

            for i in range(num_crops_h):
                for j in range(num_crops_w):
                    start_h = i * 512
                    end_h = (i + 1) * 512
                    start_w = j * 512
                    end_w = (j + 1) * 512

                    crop = cropped_image[start_h:end_h, start_w:end_w]

                    folder_name = os.path.basename(os.path.dirname(file_path))
                    output_filename = f"{folder_name}_{os.path.splitext(files[0])[0]}_crop_{i}_{j}.tiff"
                    output_file_path = os.path.join(output_path, output_filename)

                    tifffile.imwrite(output_file_path, crop)


def find_non_black_region(image):
    non_black_pixels = np.argwhere(image > 0)
    if non_black_pixels.size == 0:
        return 0, 0, image.shape[0], image.shape[1]
    top, left = non_black_pixels.min(axis=0)
    bottom, right = non_black_pixels.max(axis=0)
    return top, left, bottom, right

if __name__ == '__main__':
    
    base_path = r'\\rad-maid-001\D\Users\chloe\data\2023Lesiondetectability\Simplex projection and recon\no lesion\ForBruno_OLDGeo\Recon'
    output_path = r'D:\Users\dayv\mamografIA_upscaler\data\new_artigo_data_crop'    
    
    crop_images(base_path, output_path)

    exit()
    
    parser = argparse.ArgumentParser(description='Crop DICOM images.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing DICOM images.')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder where cropped images will be saved.')
    args = parser.parse_args()

    crop_images(args.input_folder, args.output_folder)