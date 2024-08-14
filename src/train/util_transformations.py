import pandas as pd
import numpy as np
from monai import transforms
from monai.data import CacheDataset, DataLoader
from monai.apps import MedNISTDataset
import os
import pydicom
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform
from monai.transforms import LoadImaged, EnsureChannelFirstd, CenterSpatialCropD
import torch
import cv2
from custom_transforms import ApplyTokenizerd
from lungmask import LMInferer
import SimpleITK as sitk

class LoadDICOM(MapTransform):
    """
    Custom transformation to load DICOM images with slope and intercept adjustment.
    """

    def __init__(self, keys, reader=None):
        super().__init__(keys)
        self.reader = reader or pydicom.dcmread

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            file_path = d[key]  

            dicom_data = self.reader(file_path)
            image_array = dicom_data.pixel_array

            if 'RescaleSlope' in dicom_data:
                rescale_slope = dicom_data.RescaleSlope
            else:
                rescale_slope = 1
            
            if 'RescaleIntercept' in dicom_data:
                rescale_intercept = dicom_data.RescaleIntercept
            else:
                rescale_intercept = 0
            
            image_array = image_array * rescale_slope + rescale_intercept
            w = float(dicom_data.WindowWidth)
            c = float(dicom_data.WindowCenter)

            y = np.zeros_like(image_array)
            y[image_array <= (c - 0.5 - (w - 1) / 2)] = 0
            y[image_array > (c - 0.5 + (w - 1) / 2)] = 1
            mask = (image_array > (c - 0.5 - (w - 1) / 2)) & (image_array <= (c - 0.5 + (w - 1) / 2))
            y[mask] = ((image_array[mask] - (c - 0.5)) / (w - 1) + 0.5) * (1 - 0) + 0

            image_array = np.expand_dims(y, axis=0)

            d[key] = image_array.astype(np.float32)
            d['filename'] = file_path.split('/')[-1]

        return d


class LoadDICOMmask(MapTransform):
    """
    Custom transformation to load DICOM images with slope and intercept adjustment.
    """

    def __init__(self, keys, inferer ,reader=None):
        super().__init__(keys)
        self.reader = reader or pydicom.dcmread
        self.inferer = inferer
        
    def create_lung_mask(self, image, threshold=0.8):
        """Creates a binary mask to highlight lung structures.

        Args:
            image (np.array): The windowed CT image.
            threshold (float): Relative threshold for segmentation.

        Returns:
            np.array: Binary mask with lung structures highlighted.
        """
        mask = np.zeros_like(image)
        mask[image > threshold] = 1
        return mask
    
    def create_segmentation_mask(self, image, inferer):

        input_image = sitk.ReadImage(image)
        segmentation = inferer.apply(input_image)

        segmentation = segmentation.squeeze()
        segmentation_np = np.array(segmentation)
        
        return segmentation_np

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            file_path = d[key]  

            dicom_data = self.reader(file_path)
            image_array = dicom_data.pixel_array

            if 'RescaleSlope' in dicom_data:
                rescale_slope = dicom_data.RescaleSlope
            else:
                rescale_slope = 1
            
            if 'RescaleIntercept' in dicom_data:
                rescale_intercept = dicom_data.RescaleIntercept
            else:
                rescale_intercept = 0
            
            image_array = image_array * rescale_slope + rescale_intercept
            w = float(dicom_data.WindowWidth)
            c = float(dicom_data.WindowCenter)

            y = np.zeros_like(image_array)
            y[image_array <= (c - 0.5 - (w - 1) / 2)] = 0
            y[image_array > (c - 0.5 + (w - 1) / 2)] = 1
            mask = (image_array > (c - 0.5 - (w - 1) / 2)) & (image_array <= (c - 0.5 + (w - 1) / 2))
            y[mask] = ((image_array[mask] - (c - 0.5)) / (w - 1) + 0.5) * (1 - 0) + 0

            image_array = np.expand_dims(y, axis=0)

            d[key] = image_array.astype(np.float32)
            d['filename'] = file_path.split('/')[-1]
            
            closed_lung_mask = self.create_lung_mask(y, threshold=0.8)
            segmentation_np = self.create_segmentation_mask(file_path, self.inferer)
            segmentation_np = np.where(segmentation_np >= 1, 1, 0)
            
            filtered_with_closed_lung_mask = np.where(closed_lung_mask == 1, y, 0)
            filtered_with_segmentation_np = np.where(segmentation_np == 1, y, 0)
            #image with less high pixels
            combined_filtered = np.maximum(filtered_with_closed_lung_mask, filtered_with_segmentation_np)

            lung_mask_binary = np.where(closed_lung_mask != 0, 1, 0)
            segmentation_np_mask = np.where(segmentation_np == 1, 2, 0)
            segmentation_mask_colored = np.where(filtered_with_segmentation_np >= 0.4, 4, 0)

            combined_mask = lung_mask_binary + segmentation_np_mask + segmentation_mask_colored
            
            d['mask'] = (combined_mask / combined_mask.max()).astype(np.float32)
            combined_filtered = np.expand_dims(combined_filtered, axis=0)
            d[key] = combined_filtered.astype(np.float32)
            
        return d
    
class Normalization(MapTransform):
    """
    Transformation to normalize the image by dividing each pixel by the maximum pixel value.
    """

    def __init__(self, keys: KeysCollection, min_val, max_val):
        super().__init__(keys)
        #self.max_pixel_value = max_pixel_value
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            #d[key] = d[key] / self.max_pixel_value
            d[key] = (d[key] - self.min_val) / (self.max_val - self.min_val)

        return d

class Normalization_max(MapTransform):
    """
    Transformation to normalize the image by dividing each pixel by the maximum pixel value.
    """

    def __init__(self, keys: KeysCollection, max_pixel_value):
        super().__init__(keys)
        self.max_pixel_value = max_pixel_value

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key] / self.max_pixel_value
        return d


def get_max_pixel_values(datalist):
    max_pixel_value_image = 0.0
    max_pixel_value_low_res_image = 0.0
    
    roi_image_size = 512  
    roi_low_res_size = 358
    
    load_transforms = transforms.Compose([
        LoadImaged(keys=["image", "low_res_image"], reader='PILReader'),
        EnsureChannelFirstd(keys=["image", "low_res_image"]),
        CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size, roi_image_size)),
        CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size, roi_low_res_size))
    ])
    
    for data_dict in datalist:
        loaded_data = load_transforms(data_dict)
        image = loaded_data["image"]
        low_res_image = loaded_data["low_res_image"]
        
        max_pixel_value_image = max(max_pixel_value_image, image.max())
        max_pixel_value_low_res_image = max(max_pixel_value_low_res_image, low_res_image.max())

    return max_pixel_value_image, max_pixel_value_low_res_image

def get_max_min_pixel_values(datalist):
    max_pixel_value_image = 0.0
    max_pixel_value_low_res_image = 0.0
    min_pixel_value_image = np.inf
    min_pixel_value_low_res_image = np.inf
    
    median_img = 0.0
    iqr_img = 0.0
    
    median_low_img = 0.0
    iqr_low_img = 0.0
    
    #roi_image_size = 512  
    #roi_low_res_size = 358
    
    # load_transforms = transforms.Compose([
    #     LoadImaged(keys=["image", "low_res_image"], reader='PILReader'),
    #     EnsureChannelFirstd(keys=["image", "low_res_image"]),
    #     CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size, roi_image_size)),
    #     CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size, roi_low_res_size))
    # ])
    
    load_transforms = transforms.Compose([
        #LoadImaged(keys=["image"], reader='PILReader'),
        LoadDICOM(keys=['image']),
        EnsureChannelFirstd(keys=["image"], channel_dim=0),
    ])
    
    for data_dict in datalist:
        loaded_data = load_transforms(data_dict)
        image = loaded_data["image"]
        #low_res_image = loaded_data["low_res_image"]
        
        # median_img += np.median(image)
        #iqr_img += np.percentile(image, 75) - np.percentile(image, 25)
        
        #median_low_img += np.median(low_res_image)
        #iqr_low_img += np.percentile(low_res_image, 75) - np.percentile(low_res_image, 25)
        
        max_pixel_value_image = max(max_pixel_value_image, np.max(image))
        #max_pixel_value_low_res_image = max(max_pixel_value_low_res_image, np.max(low_res_image))
        min_pixel_value_image = min(min_pixel_value_image, np.min(image))
        #min_pixel_value_low_res_image = min(min_pixel_value_low_res_image, np.min(low_res_image))
    
    #num_images = len(datalist)
    #median_img /= num_images
    #iqr_img /= num_images
    #median_low_img /= num_images
    #iqr_low_img /= num_images
    
    # dict = {'max_pixel_img': max_pixel_value_image, 'max_pixel_low_img': max_pixel_value_low_res_image,
    #         'min_pixel_img': min_pixel_value_image, 'min_pixel_low_img': min_pixel_value_low_res_image,
    #         'median_img': median_img, 'iqr_img': iqr_img, 'median_low_img': median_low_img,
    #         'iqr_low_img': iqr_low_img}
    
    dict = {'max_pixel_img': max_pixel_value_image,'min_pixel_img': min_pixel_value_image}
    
    return dict

def get_datalist(ids_path:str):
    """
    Carregamento da tabela dos caminhos para a criação de um vetor com dicionários
    para passar no dataloader específico.

    Args:
        args (str): Caminho do .tsv
    """

    df = pd.read_csv(ids_path, sep="\t")
        
    data_dicts = []
    for index, row in df.iterrows():
        # if '.tif' in row['image']:
        #     continue
        
        data_dicts.append(
            {
                "image": str(row["image"]),
                #"low_res_image": str(row['low_res_image'])
                "report": "CT image of the lungs.",
            }
        )
    print(f"{len(data_dicts)} imagens.")

    return data_dicts


def get_upsampler_dataloader(batch_size: int,training_ids: str, validation_ids: str, num_workers: int = 8):
    """
    Função que define as transformações das imagens e a criação do dataloader
    do treino e da validação.

    Args:
        batch_size (int): Tamanho do batch
        training_ids (str): Caminho do .tsv de treinamento
        validation_ids (str): Caminho do .tsv de validação
        num_workers (int): Envolve o quão rápido é feito o carregamento dos
        dados na memória (revisar)
    """
    roi_image_size = 512 # 416
    roi_low_res_size = 358 # 291
    low_res_size =  512 # 208
        
    train_datalist = get_datalist(ids_path=training_ids)[:10]
    val_datalist = get_datalist(ids_path=validation_ids)[:10]
    
    # model for segmentation of lung
    inferer = LMInferer(modelname='R231CovidWeb')

    #img_max_pixel, img_low_max_pixel = get_max_pixel_values(train_datalist) #16254 #15971
    #dict = get_max_min_pixel_values(train_datalist)
    #print(dict['max_pixel_img'])
    #print(dict['min_pixel_img'])
    train_transforms = transforms.Compose(
        [
            #transforms.LoadImaged(keys=["image", "low_res_image"], reader='PILReader'),
            #transforms.LoadImaged(keys=["image"], reader='PILReader'),
            #LoadDICOM(keys=['image']),
            LoadDICOMmask(keys=['image'], inferer=inferer),

            #transforms.EnsureChannelFirstd(keys=["image", "low_res_image"]),
            transforms.EnsureChannelFirstd(keys=["image"], channel_dim=0),
            
            #transforms.ScaleIntensityd(keys=["image", "low_res_image"], minv=0.0, maxv=1.0),
            #transforms.CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size, roi_image_size)),
            #transforms.CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size, roi_low_res_size)),
            
            Normalization(keys=["image"], min_val=0, max_val=1),
            #Normalization(keys=["low_res_image"],min_val=dict['min_pixel_low_img'], max_val=dict['max_pixel_low_img']),
            
            #Normalization_max(keys=["image"], max_pixel_value=img_max_pixel),
            #Normalization_max(keys=["low_res_image"], max_pixel_value=img_low_max_pixel),
            # transforms.RandFlipd( #
            #     keys=["image"],#, "low_res_image"
            #     spatial_axis=0,
            #     prob=0.2,
            # ),
            # transforms.RandFlipd( #
            #     keys=["image"],#, "low_res_image"
            #     spatial_axis=1,
            #     prob=0.2,
            # ),
            # transforms.RandRotate90d(
            #     keys=["image"],#, "low_res_image"
            #     prob=0.2,
            # ),
            
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(keys=["low_res_image"],spatial_size=(low_res_size, low_res_size)),
            transforms.ThresholdIntensityd(keys=["image","low_res_image"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["image","low_res_image"], threshold=0, above=True, cval=0),
            #ApplyTokenizerd(keys=["report"]),
            transforms.ToTensord(keys=["image",'low_res_image','mask']) 
        ]
    )
    
    val_transforms = transforms.Compose(
        [
            #transforms.LoadImaged(keys=["image", "low_res_image"], reader='PILReader'),
            #transforms.LoadImaged(keys=["image"], reader='PILReader'),
            #LoadDICOM(keys=['image']),
            LoadDICOMmask(keys=['image'], inferer=inferer),
            
            #transforms.EnsureChannelFirstd(keys=["image", "low_res_image"]),
            transforms.EnsureChannelFirstd(keys=["image"], channel_dim=0),
            
            #transforms.ScaleIntensityd(keys=["image", "low_res_image"], minv=0.0, maxv=1.0),
            #transforms.CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size, roi_image_size)),
            #transforms.CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size, roi_low_res_size)),
            
            Normalization(keys=["image"], min_val=0, max_val=1),
            #Normalization(keys=["low_res_image"],min_val=dict['min_pixel_low_img'], max_val=dict['max_pixel_low_img']),
            
            #ormalization_max(keys=["image"], max_pixel_value=img_max_pixel),
            #Normalization_max(keys=["low_res_image"], max_pixel_value=img_low_max_pixel),
    
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(keys=["low_res_image"],spatial_size=(low_res_size, low_res_size)),
            transforms.ThresholdIntensityd(keys=["image","low_res_image"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["image","low_res_image"], threshold=0, above=True, cval=0),
            #ApplyTokenizerd(keys=["report"]),
            transforms.ToTensord(keys=["image",'low_res_image','mask']) 
            
        ]
    )

    #train_datalist = get_datalist(ids_path=training_ids)
    train_ds = CacheDataset(data=train_datalist, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=True)

    #val_datalist = get_datalist(ids_path=validation_ids)
    val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)

    return train_loader, val_loader

def get_upsampler_dataloader_without_low_res(batch_size: int,training_ids: str, validation_ids: str, num_workers: int = 8):
    """
    Função que define as transformações das imagens e a criação do dataloader
    do treino e da validação.

    Args:
        batch_size (int): Tamanho do batch
        training_ids (str): Caminho do .tsv de treinamento
        validation_ids (str): Caminho do .tsv de validação
        num_workers (int): Envolve o quão rápido é feito o carregamento dos
        dados na memória (revisar)
    """
    roi_image_size = 416

    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size,roi_image_size)),
            transforms.ToTensord(keys=["image"]) 
        ]
    )
    
    val_transforms = transforms.Compose(
        [
          transforms.LoadImaged(keys=["image"]),
          transforms.EnsureChannelFirstd(keys=["image"]),
          transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
          transforms.CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size,roi_image_size)),
          transforms.ToTensord(keys=["image"]),
        ]
    )

    train_datalist = get_datalist(ids_path=training_ids)
    train_ds = CacheDataset(data=train_datalist, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=True)

    val_datalist = get_datalist(ids_path=validation_ids)
    val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)

    return train_loader, val_loader


def get_upsampler_dataloader_mednist(batch_size: int,training_ids: str, validation_ids: str, num_workers: int = 8):
    """
    Função que define as transformações das imagens e a criação do dataloader
    do treino e da validação.

    Args:
        batch_size (int): Tamanho do batch
        training_ids (str): Caminho do .tsv de treinamento
        validation_ids (str): Caminho do .tsv de validação
        num_workers (int): Envolve o quão rápido é feito o carregamento dos
        dados na memória (revisar)
    """
    path_root = '/project/sr_data_from_tutorial'
    os.makedirs(path_root, exist_ok=True)
    image_size = 64

    # train_transforms = transforms.Compose(
    #     [
    #         transforms.LoadImaged(keys=["image"]),
    #         transforms.EnsureChannelFirstd(keys=["image"]),
     #         transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    #         transforms.CenterSpatialCropD(keys=["image"], roi_size=(image_size,image_size)),
    #         transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
    #         transforms.Resized(keys=["low_res_image"], spatial_size=(16, 16)),
    #         transforms.ToTensord(keys=["image",'low_res_image']),
    #     ]
    # )
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.RandAffined(
                keys=["image"],
                rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
                translate_range=[(-1, 1), (-1, 1)],
                scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                spatial_size=[image_size, image_size],
                padding_mode="zeros",
                prob=0.5,
            ),
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(keys=["low_res_image"], spatial_size=(16, 16)),
        ]
    )
    # val_transforms = transforms.Compose(
    #     [
    #       transforms.LoadImaged(keys=["image"]),
    #       transforms.EnsureChannelFirstd(keys=["image"]),
    #       transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    #       transforms.CenterSpatialCropD(keys=["image"], roi_size=(image_size,image_size)),
    #       transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
    #       transforms.Resized(keys=["low_res_image"], spatial_size=(16, 16)),
    #       transforms.ToTensord(keys=["image",'low_res_image'])
    #     ]
    # )
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(keys=["low_res_image"], spatial_size=(16, 16)),
        ]
    )
    
    train_data = MedNISTDataset(root_dir=path_root, section="training", download=False, seed=0)
    train_datalist_ = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "HeadCT" ]
    train_ds = CacheDataset(data=train_datalist_, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)

    val_data = MedNISTDataset(root_dir=path_root, section="validation", download=False, seed=0)
    val_datalist_ = [{"image": item["image"]} for item in val_data.data if item["class_name"] == "HeadCT"]
    val_ds = CacheDataset(data=val_datalist_, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader


def get_upsampler_dataloader_mednist_2dldm(batch_size: int,training_ids: str, validation_ids: str, num_workers: int = 8,):
    """
    Função que define as transformações das imagens e a criação do dataloader
    do treino e da validação.

    Args:
        batch_size (int): Tamanho do batch
        training_ids (str): Caminho do .tsv de treinamento
        validation_ids (str): Caminho do .tsv de validação
        num_workers (int): Envolve o quão rápido é feito o carregamento dos
        dados na memória (revisar)
    """
    path_root = '/project/sr_data_from_tutorial'
    os.makedirs(path_root, exist_ok=True)
    image_size = 64

    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.CenterSpatialCropD(keys=["image"], roi_size=(image_size,image_size)),
            transforms.ToTensord(keys=["image"]),

        ]
    )

    val_transforms = transforms.Compose(
        [
          transforms.LoadImaged(keys=["image"]),
          transforms.EnsureChannelFirstd(keys=["image"]),
          transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
          transforms.CenterSpatialCropD(keys=["image"], roi_size=(image_size,image_size)),
          transforms.ToTensord(keys=["image"])
        ]
    )
    
    train_data = MedNISTDataset(root_dir=path_root, section="training", download=False, seed=0)
    train_datalist_ = [{"image": item["image"]} for item in train_data.data if item["class_name"] == "Hand"]
    train_ds = CacheDataset(data=train_datalist_, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)

    val_data = MedNISTDataset(root_dir=path_root, section="validation", download=False, seed=0)
    val_datalist_ = [{"image": item["image"]} for item in val_data.data if item["class_name"] == "Hand"]
    val_ds = CacheDataset(data=val_datalist_, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader