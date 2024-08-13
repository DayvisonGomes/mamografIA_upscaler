import os
import tifffile
import numpy as np

if __name__ == '__main__':
    
    folder_path = r'D:\Users\dayv\mamografIA_upscaler\outputs\recons_imgs_valid_clinical_franklin'
    files = os.listdir(folder_path)
    output_dir = r'D:\Users\dayv\mamografIA_upscaler\outputs\recons_imgs_valid_clinical_franklin_16bits'
    os.makedirs(output_dir, exist_ok=True)
    
    max_pixel_value = 4095
    min_pixel_value = 0
    
    for file in files:
        if file.endswith('.tif') or file.endswith('.tiff'):
            file_path = os.path.join(folder_path, file)
            img = tifffile.imread(file_path)
            desnormalized_image = img * (max_pixel_value - min_pixel_value) + min_pixel_value
            desnormalized_image = np.clip(desnormalized_image, min_pixel_value, max_pixel_value)
            desnormalized_image = desnormalized_image.astype(np.uint16)
            
            if '.tiff_recons' in file:
                
                path_save = os.path.join(output_dir, file.split('.tiff_recons')[0] + '_recons.tiff')
            
            else:
                path_save = os.path.join(output_dir, file.split('.tiff')[0] + '.tiff')
            
            tifffile.imwrite(path_save, desnormalized_image)

