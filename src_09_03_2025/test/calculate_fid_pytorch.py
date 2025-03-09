from torchmetrics.image.fid import FrechetInceptionDistance
import tifffile
import os
import torch
import numpy as np
from scipy.stats import wasserstein_distance

def calculate_fid():
    path_ = '/project/outputs/upscale_valid_set_final_with_interpolate'
    path_files = os.listdir(path_)

    real_images = torch.tensor([])
    fake_images = torch.tensor([])

    for sample_path in sorted(path_files):
        if sample_path == 'logs':
            continue
        if 'upscale' in sample_path and 'upscale_interpolate' not in sample_path:
            img_np = tifffile.imread(os.path.join(path_, sample_path))
            fake_images = torch.cat([fake_images, torch.tensor(img_np).unsqueeze(0)], dim=0)
            
        elif 'upscale' not in sample_path and 'upscale_interpolate' not in sample_path:
            img_np = tifffile.imread(os.path.join(path_, sample_path))
            real_images = torch.cat([real_images, torch.tensor(img_np).unsqueeze(0)], dim=0)

    real_images = real_images.unsqueeze(1)
    fake_images = fake_images.unsqueeze(1)

    real_images_rgb = torch.cat([real_images] * 3, dim=1)
    fake_images_rgb = torch.cat([fake_images] * 3, dim=1)

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images_rgb, real=True)
    fid.update(fake_images_rgb, real=False)

    print(f"FID: {float(fid.compute())}")

def calculate_wasserstein():
    path = '/project/outputs/runs_final/generate_test_4latent_pulmao_exp_4_sr_last'
    #path = '/project/outputs/runs_final/generate_test_4latent_pulmao_exp_3'
    files = [f for f in os.listdir(path) if '_' not in f and 'logs' not in f]

    all_images = []
    
    for file in files:
        original_img = tifffile.imread(os.path.join(path, file))
        img_name = file.split('.')[0]
        gen_img = tifffile.imread(os.path.join(path, f'{img_name}_generate.tiff'))
        all_images.append(original_img)
        all_images.append(gen_img)

    global_min = min(img.min() for img in all_images)
    global_max = max(img.max() for img in all_images)
    
    bins = 256
    wasserstein_distances = {}

    print(f"Intervalo global: [{global_min}, {global_max}]")

    for file in files:
        original_img = tifffile.imread(os.path.join(path, file))
        img_name = file.split('.')[0]

        gen_img = tifffile.imread(os.path.join(path, f'{img_name}_generate.tiff'))
        
        hist1, bin_edges = np.histogram(original_img, bins=bins, range=(global_min, global_max), density=True)
        hist2, _ = np.histogram(gen_img, bins=bins, range=(global_min, global_max), density=True)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

        dist_wasserstein = wasserstein_distance(bin_centers, bin_centers, hist1, hist2)
    
        wasserstein_distances[file] = dist_wasserstein
        print(f"{file}: {dist_wasserstein}")

    print('Wasserstein distances mean:', round(np.mean(list(wasserstein_distances.values())), 4))
    print('In folder:', path)
    return wasserstein_distances

if __name__ == '__main__':
    dist = calculate_wasserstein()    