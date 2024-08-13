from torchmetrics.image.fid import FrechetInceptionDistance
import tifffile
import os
import torch

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
