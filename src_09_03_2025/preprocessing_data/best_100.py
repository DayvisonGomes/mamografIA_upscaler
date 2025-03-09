import csv
from generative.metrics import SSIMMetric, MultiScaleSSIMMetric
import os
import tifffile
import numpy as np
import torch
from decimal import Decimal

if __name__ == '__main__':
    mssim_metric = MultiScaleSSIMMetric(spatial_dims=2, kernel_size=7)

    path_imgs = '/project/outputs/runs_final/upscale_valid_3latent_pulmao_3'
    file_names = os.listdir(path_imgs)

    dict_img_mssim = {}
    
    for file in file_names:
        if file == 'logs':
            continue
        
        if 'upscale' in file:
            continue
        
        name = file.split('.tiff')[0]
        original_file = os.path.join(path_imgs, file)
        upscale_file = os.path.join(path_imgs, name + '_upscale.tiff')

        original_img = tifffile.imread(original_file)
        upscale_img = tifffile.imread(upscale_file)
        
        original_img = np.expand_dims(original_img, axis=0)
        original_img = np.expand_dims(original_img, axis=0)

        upscale_img = np.expand_dims(upscale_img, axis=0)
        upscale_img = np.expand_dims(upscale_img, axis=0)

        mssim_value = mssim_metric(torch.tensor(original_img), torch.tensor(upscale_img))[0, 0].item()
        print(mssim_value)
        dict_img_mssim[original_file] = mssim_value
    
    sorted_dict = dict(sorted(dict_img_mssim.items(), key=lambda item: item[1], reverse=True))
    
    with open('/project/mssim_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['original_file', 'mssim'])
        for original_file, mssim in sorted_dict.items():
            writer.writerow([original_file, f"{mssim:.15f}"])
