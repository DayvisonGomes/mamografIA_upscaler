import argparse
import torch
import torch.nn.functional as F
from generative.metrics import FIDMetric
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--sample_dir", help="Location of the samples to evaluate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    set_determinism(seed=args.seed)
    print_config()

    samples_dir = os.listdir(args.sample_dir)
    
    # Load pretrained model
    device = torch.device("cuda")
    model = torch.hub.load("Warvito/MedicalNet-models", "medicalnet_resnet50_23datasets")
    model = model.to(device)
    model.eval()

    # Samples
    samples_datalist = []
    valid_datalist = []

    for sample_path in sorted(samples_dir):
        if sample_path == 'logs':
            continue
        if '_upscale' in sample_path and 'upscale_interpolate' not in sample_path:
            samples_datalist.append(
                {
                    "image": os.path.join(args.sample_dir, str(sample_path)),
                }
            )
        elif 'upscale' not in sample_path and 'upscale_interpolate' not in sample_path:
            valid_datalist.append(
                {
                    "image": os.path.join(args.sample_dir, str(sample_path)),
                }
            )
            
    print(f"{len(samples_datalist)} images found in {str(args.sample_dir)}")
    print(f"{len(valid_datalist)} images found in {str(args.sample_dir)}")

    sample_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], reader='PILReader'),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    samples_ds = Dataset(
        data=samples_datalist,
        transform=sample_transforms,
    )
    samples_loader = DataLoader(
        samples_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    samples_features = []
    for batch in tqdm(samples_loader):
        img = batch["image"]
        with torch.no_grad():
            outputs = model(img.unsqueeze(0).to(device))
            outputs = F.adaptive_avg_pool2d(outputs, (1, 1)).view(outputs.size(0), -1)  # Global average pooling
            
        samples_features.append(outputs.cpu())
    samples_features = torch.cat(samples_features, dim=0)

    valid_ds = Dataset(data=valid_datalist, transform=sample_transforms)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    test_features = []
    for batch in tqdm(valid_loader):
        img = batch["image"]
        with torch.no_grad():
            outputs = model(img.unsqueeze(0).to(device))
            outputs = F.adaptive_avg_pool2d(outputs, (1, 1)).view(outputs.size(0), -1)  # Global average pooling

        test_features.append(outputs.cpu())
    test_features = torch.cat(test_features, dim=0)

    print(samples_features.shape)
    print(test_features.shape)
    # Compute FID
    metric = FIDMetric()
    fid = metric(test_features, samples_features)

    print(fid.item())