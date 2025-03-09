import argparse
import os
import numpy as np
import pandas as pd
import torch
import tifffile
from generative.metrics import SSIMMetric, MultiScaleSSIMMetric
from monai.metrics import MAEMetric, PSNRMetric

def reshape_masked_image(original, mask, value):
    reshaped = torch.zeros_like(torch.tensor(original, dtype=torch.float32))
    reshaped[mask == value] = torch.tensor(original[mask == value], dtype=torch.float32)
    return reshaped.unsqueeze(0).unsqueeze(0)

if __name__ == "__main__":
    path = "/project/generate_test_4latent_pulmao_exp_4_sr_last"
    files = os.listdir(path)
    
    img_gen_paths = []
    img_original_paths = []
    img_mask_paths = []

    psnr_metric = PSNRMetric(max_val=1.0)
    mae_metric = MAEMetric()
    mssim_metric = MultiScaleSSIMMetric(spatial_dims=2, kernel_size=7)
    ssim_metric = SSIMMetric(spatial_dims=2, kernel_size=7)

    psnr_total_lung = 0
    mae_total_lung = 0
    mssim_total_lung = 0
    
    psnr_total_tissue = 0
    mae_total_tissue = 0
    mssim_total_tissue = 0

    quant_imgs = 0

    for file in files:
        if file == 'logs':
            continue
        
        quant_imgs += 1

        if '_generate' in file:
            img_gen_paths.append(file)

        if '_' not in file:
            img_original_paths.append(file)

        if '_mask' in file:
            img_mask_paths.append(file)

    for i in range(len(img_gen_paths)):
        img_gen = img_gen_paths[i]
        img_mask = img_mask_paths[i]
        img_original = img_original_paths[i]

        img_mask = tifffile.imread(os.path.join(path, img_mask))
        img_gen = tifffile.imread(os.path.join(path, img_gen))
        img_original = tifffile.imread(os.path.join(path, img_original))

        if 1 not in img_mask:
            img_gen_filter_tissue = reshape_masked_image(img_gen, img_mask, 0.5)
            img_original_filter_tissue = reshape_masked_image(img_original, img_mask, 0.5)
            
            psnr_value_tissue = psnr_metric(img_original_filter_tissue, img_gen_filter_tissue)
            mae_value_tissue = mae_metric(img_original_filter_tissue, img_gen_filter_tissue)
            mssim_value_tissue = mssim_metric(img_original_filter_tissue, img_gen_filter_tissue)

            psnr_total_tissue += psnr_value_tissue[0, 0].item()
            mae_total_tissue += mae_value_tissue[0, 0].item()
            mssim_total_tissue += mssim_value_tissue[0, 0].item()

        else:   
            img_gen_filter_lung = reshape_masked_image(img_gen, img_mask, 1)
            img_gen_filter_tissue = reshape_masked_image(img_gen, img_mask, 0.5)
            img_original_filter_lung = reshape_masked_image(img_original, img_mask, 1)
            img_original_filter_tissue = reshape_masked_image(img_original, img_mask, 0.5)

            psnr_value_lung = psnr_metric(img_original_filter_lung, img_gen_filter_lung)
            mae_value_lung = mae_metric(img_original_filter_lung, img_gen_filter_lung)
            mssim_value_lung = mssim_metric(img_original_filter_lung, img_gen_filter_lung)

            psnr_value_tissue = psnr_metric(img_original_filter_tissue, img_gen_filter_tissue)
            mae_value_tissue = mae_metric(img_original_filter_tissue, img_gen_filter_tissue)
            mssim_value_tissue = mssim_metric(img_original_filter_tissue, img_gen_filter_tissue)

            psnr_total_lung += psnr_value_lung[0, 0].item()
            mae_total_lung += mae_value_lung[0, 0].item()
            mssim_total_lung += mssim_value_lung[0, 0].item()

            psnr_total_tissue += psnr_value_tissue[0, 0].item()
            mae_total_tissue += mae_value_tissue[0, 0].item()
            mssim_total_tissue += mssim_value_tissue[0, 0].item()

    print('MS-SSIM lung region: ', mssim_total_lung / quant_imgs)
    print('PSNR lung region: ', psnr_total_lung / quant_imgs)
    print('MAE lung region: ', mae_total_lung / quant_imgs)

    print('MS-SSIM tissue region: ', mssim_total_tissue / quant_imgs)
    print('PSNR tissue region: ', psnr_total_tissue / quant_imgs)
    print('MAE tissue region: ', mae_total_tissue / quant_imgs)

    """
    MS-SSIM lung region:  0.09787368112140232
    PSNR lung region:  3.39219175974528
    MAE lung region:  0.0006148384373065913
    MS-SSIM tissue region:  0.100937686363856
    PSNR tissue region:  3.0218055513170032
    MAE tissue region:  0.001620865354521407
    """