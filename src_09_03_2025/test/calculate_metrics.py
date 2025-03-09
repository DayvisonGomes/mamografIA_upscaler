import argparse
import os
import numpy as np
from generative.metrics import MultiScaleSSIMMetric
from monai import transforms
from monai.config import print_config
from monai.metrics import MAEMetric, PSNRMetric
from monai.utils import set_determinism
from tqdm import tqdm
import pandas as pd
from monai import transforms
from monai.data import Dataset
from torch.utils.data import DataLoader

def get_datalist(
    ids_path: str,
    upper_limit: int | None = None,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")

    if upper_limit is not None:
        df = df[:upper_limit]

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "image": str(row["image"]),
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts

def get_test_dataloader(
    batch_size: int,
    test_ids: str,
    num_workers: int = 8,
    upper_limit: int | None = None,
):
    image_size = 732 #*
    low_res_size = 512
    
    test_transforms = transforms.Compose(
        [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
        transforms.Resized(keys=["image"], spatial_size=(image_size, image_size)),
        transforms.Resized(keys=["low_res_image"], spatial_size=(low_res_size, low_res_size)),
        transforms.ToTensord(keys=["image", "low_res_image"]),
        ]
    )

    test_dicts = get_datalist(ids_path=test_ids, upper_limit=upper_limit)
    test_ds = Dataset(data=test_dicts, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return test_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--samples_dir", help="Location of the samples to evaluate.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    set_determinism(seed=args.seed)
    print_config()

    samples_dir = os.listdir(args.samples_dir)

    sample_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    # Test set
    test_loader = get_test_dataloader(
        batch_size=1,
        test_ids=args.test_ids,
        num_workers=args.num_workers,
        upper_limit=1000,
    )

    psnr_metric = PSNRMetric(max_val=1.0)
    mae_metric = MAEMetric()
    mssim_metric = MultiScaleSSIMMetric(spatial_dims=2, kernel_size=7)

    psnr_list = []
    mae_list = []
    mssim_list = []
    for batch in tqdm(test_loader):
        img = batch["image"]
        img_synthetic = sample_transforms(
            {"image": os.path.join(samples_dir, f"{batch['image_meta_dict']['filename_or_obj'][0]}_upscaled.png")  }
        )["image"].unsqueeze(1)

        psnr_value = psnr_metric(img, img_synthetic)
        mae_value = mae_metric(img, img_synthetic)
        mssim_value = mssim_metric(img, img_synthetic)

        psnr_list.append(psnr_value.item())
        mae_list.append(mae_value.item())
        mssim_list.append(mssim_value.item())

    print(f"PSNR: {np.mean(psnr_list)}+-{np.std(psnr_list)}")
    print(f"MAE: {np.mean(mae_list)}+-{np.std(mae_list)}")
    print(f"MSSIM: {np.mean(mssim_list)}+-{np.std(mssim_list)}")
