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
from monai.data import PersistentDataset
import scipy.ndimage as ndimage
import tifffile
from neuroHarmonize import harmonizationLearn, harmonizationApply

# model for segmentation of lung
#inferer = LMInferer(modelname='R231CovidWeb')

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

            # y = np.zeros_like(image_array)
            # y[image_array <= (c - 0.5 - (w - 1) / 2)] = 0
            # y[image_array > (c - 0.5 + (w - 1) / 2)] = 1
            # mask = (image_array > (c - 0.5 - (w - 1) / 2)) & (image_array <= (c - 0.5 + (w - 1) / 2))
            # y[mask] = ((image_array[mask] - (c - 0.5)) / (w - 1) + 0.5) * (1 - 0) + 0

            #image_array = np.expand_dims(y, axis=0)
            image_array = np.expand_dims(image_array, axis=0)

            d[key] = image_array.astype(np.float32)
            d[f'filename_{key}'] = file_path.split('/')[-1]

        return d


class LoadDICOMmask(MapTransform):
    """
    Custom transformation to load DICOM images with slope and intercept adjustment.
    """

    def __init__(self, keys ,reader=None):
        super().__init__(keys)
        self.reader = reader or pydicom.dcmread
        
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
    
    def fill_holes_in_mask(self, mask):
        filled_mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        return filled_mask
    
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

            #image_array = np.expand_dims(y, axis=0)

            #d[key] = image_array.astype(np.float32)
            d['filename'] = file_path.split('/')[-1]
            mask_path = file_path.split('/')[-1] + '_mask.tiff'
            mask_img = tifffile.imread(os.path.join('/project/data_lung_multiclass_masks', mask_path))
            
            combined_filtered = np.where(mask_img >= 0.5, y, 0)
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

class Normalization_std(MapTransform):

    def __init__(self, keys: KeysCollection, mean, std):
        super().__init__(keys)
        self.mean = mean
        self.std = std

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = (d[key] - self.mean) / self.std
        return d

class CombatNormalization:
    """
    Transformação personalizada para aplicar a normalização Combat nas imagens.
    """
    def __init__(self, keys, batch_col="batch"):
        self.keys = keys
        self.batch_col = batch_col
        self.combat_models = {}
    
    def fit(self, dataset):
        """Aprende os parâmetros do Combat a partir do conjunto de treinamento."""
        batch_labels = np.array([sample[self.batch_col] for sample in dataset])

        for key in self.keys:
            data = np.array([sample[key].flatten() for sample in dataset])  # Flatten todas as imagens para vetores
            self.combat_models[key] = harmonizationLearn(data, batch_labels)
    
    def __call__(self, data):
        """Aplica a normalização Combat nos dados."""
        batch_label = np.array([sample[self.batch_col] for sample in data])  # Coleta os rótulos dos batches

        for key in self.keys:
            if key in self.combat_models:
                # Coletar todas as imagens para aplicar normalização em batch
                images = np.array([sample[key].flatten() for sample in data])
                norm_images = harmonizationApply(images, self.combat_models[key], batch_label)

                # Retorna ao formato original
                for i, sample in enumerate(data):
                    sample[key] = norm_images[i].reshape(sample[key].shape)

        return data


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
    
    load_transforms = transforms.Compose([
        #LoadImaged(keys=["image", "low_res_image"]),
        LoadDICOM(keys=['image','low_res_image']),
        EnsureChannelFirstd(keys=["image", "low_res_image"], channel_dim=0),
        #CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size, roi_image_size)),
        #CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size, roi_low_res_size))
    ])
    
    # load_transforms = transforms.Compose([
    #     #LoadImaged(keys=["image"], reader='PILReader'),
    #     LoadDICOM(keys=['image']),
    #     EnsureChannelFirstd(keys=["image"], channel_dim=0),
    # ])
    
    for data_dict in datalist:
        loaded_data = load_transforms(data_dict)
        image = loaded_data["image"]
        low_res_image = loaded_data["low_res_image"]
        
        # median_img += np.median(image)
        #iqr_img += np.percentile(image, 75) - np.percentile(image, 25)
        
        #median_low_img += np.median(low_res_image)
        #iqr_low_img += np.percentile(low_res_image, 75) - np.percentile(low_res_image, 25)
        
        max_pixel_value_image = max(max_pixel_value_image, np.max(image))
        max_pixel_value_low_res_image = max(max_pixel_value_low_res_image, np.max(low_res_image))
        min_pixel_value_image = min(min_pixel_value_image, np.min(image))
        min_pixel_value_low_res_image = min(min_pixel_value_low_res_image, np.min(low_res_image))
    
    #num_images = len(datalist)
    #median_img /= num_images
    #iqr_img /= num_images
    #median_low_img /= num_images
    #iqr_low_img /= num_images
    
    # dict = {'max_pixel_img': max_pixel_value_image, 'max_pixel_low_img': max_pixel_value_low_res_image,
    #         'min_pixel_img': min_pixel_value_image, 'min_pixel_low_img': min_pixel_value_low_res_image,
    #         'median_img': median_img, 'iqr_img': iqr_img, 'median_low_img': median_low_img,
    #         'iqr_low_img': iqr_low_img}
    
    dict = {'max_pixel_img': max_pixel_value_image,'min_pixel_img': min_pixel_value_image, 'max_pixel_low_img':max_pixel_value_low_res_image,
            'min_pixel_low_img':min_pixel_value_low_res_image}
    
    return dict

def get_mean_std_pixel_values(datalist):
    mean_values_image = []
    mean_values_low_res_image = []
    mean_values_image_vmi_31 = []
    mean_values_image_lumen_36 = []
    mean_values_image_iodine_38 = []

    std_values_image = []
    std_values_low_res_image = []
    std_values_image_vmi_31 = []
    std_values_image_lumen_36 = []
    std_values_image_iodine_38 = []

    load_transforms = transforms.Compose([
        LoadDICOM(keys=['image', 'low_res_image','image_vmi_31','image_lumen_36','image_iodine_38']),
        EnsureChannelFirstd(keys=["image", "low_res_image",'image_vmi_31','image_lumen_36','image_iodine_38'], channel_dim=0)
    ])
    
    for data_dict in datalist:
        loaded_data = load_transforms(data_dict)
        image = loaded_data["image"]
        low_res_image = loaded_data["low_res_image"]
        image_vmi_31 = loaded_data["image_vmi_31"]
        image_lumen_36 = loaded_data["image_lumen_36"]
        image_iodine_38 = loaded_data["image_iodine_38"]

        mean_image = np.mean(image)
        mean_low_res_image = np.mean(low_res_image)
        mean_image_vmi_31 = np.mean(image_vmi_31)
        mean_image_lumen_36 = np.mean(image_lumen_36)
        mean_image_iodine_38 = np.mean(image_iodine_38)

        std_image = np.std(image)
        std_low_res_image = np.std(low_res_image)
        std_image_vmi_31 = np.std(image_vmi_31)
        std_image_lumen_36 = np.std(image_lumen_36)
        std_image_iodine_38 = np.std(image_iodine_38)

        mean_values_image.append(mean_image)
        mean_values_low_res_image.append(mean_low_res_image)
        mean_values_image_vmi_31.append(mean_image_vmi_31)
        mean_values_image_lumen_36.append(mean_image_lumen_36)
        mean_values_image_iodine_38.append(mean_image_iodine_38)

        std_values_image.append(std_image)
        std_values_low_res_image.append(std_low_res_image)
        std_values_image_vmi_31.append(std_image_vmi_31)
        std_values_image_lumen_36.append(std_image_lumen_36)
        std_values_image_iodine_38.append(std_image_iodine_38)

    mean_values_image = np.array(mean_values_image)
    mean_values_low_res_image = np.array(mean_values_low_res_image)
    mean_values_image_vmi_31 = np.array(mean_values_image_vmi_31)
    mean_values_image_lumen_36 = np.array(mean_values_image_lumen_36)
    mean_values_image_iodine_38 = np.array(mean_values_image_iodine_38)

    std_values_image = np.array(std_values_image)
    std_values_low_res_image = np.array(std_values_low_res_image)
    std_values_image_vmi_31 = np.array(std_values_image_vmi_31)
    std_values_image_lumen_36 = np.array(std_values_image_lumen_36)
    std_values_image_iodine_38 = np.array(std_values_image_iodine_38)
    
    overall_mean_image = np.mean(mean_values_image)
    overall_mean_low_res_image = np.mean(mean_values_low_res_image)
    overall_mean_image_vmi_31 = np.mean(mean_values_image_vmi_31)
    overall_mean_image_lumen_36 = np.mean(mean_values_image_lumen_36)
    overall_mean_image_iodine_38 = np.mean(mean_values_image_iodine_38)

    overall_std_image = np.mean(std_values_image)  
    overall_std_low_res_image = np.mean(std_values_low_res_image) 
    overall_std_image_vmi_31 = np.mean(std_values_image_vmi_31) 
    overall_std_image_lumen_36 = np.mean(std_values_image_lumen_36)
    overall_std_image_iodine_38 = np.mean(std_values_image_iodine_38)

    
    dict = {
        'mean_pixel_img': overall_mean_image,
        'std_pixel_img': overall_std_image,
        'mean_pixel_low_img': overall_mean_low_res_image,
        'std_pixel_low_img': overall_std_low_res_image,
        'mean_pixel_img_vmi_31': overall_mean_image_vmi_31,
        'std_pixel_img_vmi_31': overall_std_image_vmi_31,
        'mean_pixel_img_lumen_36': overall_mean_image_lumen_36,
        'std_pixel_img_lumen_36': overall_std_image_lumen_36,
        'mean_pixel_img_iodine_38': overall_mean_image_iodine_38,
        'std_pixel_img_iodine_38': overall_std_image_iodine_38
    }
    
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
                "image": str(row["image_high"]),
                "low_res_image": str(row['image_low']),
                "image_vmi_31": str(row['image_vmi_31']),
                "image_lumen_36": str(row['image_lumen_36']),
                "image_iodine_38": str(row['image_iodine_38']),
                "report": "CT image of the lungs.",
            }
        )
    print(f"{len(data_dicts)} imagens.")

    return data_dicts

def get_upsampler_dataloader_combat(batch_size: int, training_ids: str, validation_ids: str, num_workers: int = 8):
    roi_image_size = 512
    low_res_size = 256

    train_datalist = get_datalist(training_ids)
    val_datalist = get_datalist(validation_ids)
    
    #combat_norm = CombatNormalization(keys=["image", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"], batch_col="batch")
    #combat_norm.fit(train_datalist)

    train_transforms = transforms.Compose([
        LoadDICOM(keys=["image", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"]),
        transforms.EnsureChannelFirstd(keys=["image", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"], channel_dim=0),
        #combat_norm,
        Normalization(keys=["image"], min_val=-1803.0, max_val=23771.0),
        Normalization(keys=["low_res_image"], min_val=-1803.0, max_val=23771.0),
        Normalization(keys=["image_vmi_31"], min_val=-1803.0, max_val=23771.0),
        Normalization(keys=["image_lumen_36"], min_val=-1803.0, max_val=23771.0),
        Normalization(keys=["image_iodine_38"], min_val=-1803.0, max_val=23771.0),

        #transforms.ScaleIntensityd(keys=["image", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"], minv=0.0, maxv=1.0),
        transforms.Resized(keys=["low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"], spatial_size=(low_res_size, low_res_size)),
        transforms.ThresholdIntensityd(keys=["image", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"], threshold=1, above=False, cval=1.0),
        transforms.ThresholdIntensityd(keys=["image", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"], threshold=0, above=True, cval=0),
        transforms.ToTensord(keys=["image", "report", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"])
    ])

    val_transforms = transforms.Compose([
        LoadDICOM(keys=["image", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"]),
        transforms.EnsureChannelFirstd(keys=["image", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"], channel_dim=0),
        #combat_norm,
        Normalization(keys=["image"], min_val=-1803.0, max_val=23771.0),
        Normalization(keys=["low_res_image"], min_val=-1803.0, max_val=23771.0),
        Normalization(keys=["image_vmi_31"], min_val=-1803.0, max_val=23771.0),
        Normalization(keys=["image_lumen_36"], min_val=-1803.0, max_val=23771.0),
        Normalization(keys=["image_iodine_38"], min_val=-1803.0, max_val=23771.0),

        #transforms.ScaleIntensityd(keys=["image", "low_res_image"], minv=0.0, maxv=1.0),
        transforms.Resized(keys=["low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"], spatial_size=(low_res_size, low_res_size)),
        transforms.ThresholdIntensityd(keys=["image", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"], threshold=1, above=False, cval=1.0),
        transforms.ThresholdIntensityd(keys=["image", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"], threshold=0, above=True, cval=0),
        transforms.ToTensord(keys=["image", "report", "low_res_image", "image_vmi_31", "image_lumen_36", "image_iodine_38"])
    ])

    train_ds = CacheDataset(data=train_datalist, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)

    val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


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
    low_res_size =  256 # 208
        
    train_datalist = get_datalist(ids_path=training_ids)
    val_datalist = get_datalist(ids_path=validation_ids)

    #img_max_pixel, img_low_max_pixel = get_max_pixel_values(train_datalist) #16254 #15971
    #dict = get_max_min_pixel_values(train_datalist)
    #dict = get_mean_std_pixel_values(train_datalist)
    
    #print(dict['mean_pixel_img']) # -202.67238, -248.17827 -231.32724
    #print(dict['std_pixel_img']) # 267.81094, 351.01398 335.3502
    #print(dict['mean_pixel_low_img']) #-190.22307, 5.9611793 -231.2267
    #print(dict['std_pixel_low_img']) #266.4366, 26.424929 329.37555

    # print(dict['max_pixel_img']) #22144.0 , 24326.0
    # print(dict['max_pixel_low_img']) # 24326.0 , 5075.0
    # print(dict['min_pixel_img']) # -1924.0 , -1796.0
    # print(dict['min_pixel_low_img']) # -3528.0, -217.0
    
    train_transforms = transforms.Compose(
        [   
            #transforms.LoadImaged(keys=["image", "low_res_image"]),
            #transforms.LoadImaged(keys=["image"], reader='PILReader'),
            LoadDICOM(keys=['image','low_res_image','image_vmi_31','image_lumen_36','image_iodine_38']),
            #LoadDICOMmask(keys=['image']),

            transforms.EnsureChannelFirstd(keys=["image", 'low_res_image','image_vmi_31','image_lumen_36','image_iodine_38'], channel_dim=0),
            #ransforms.EnsureChannelFirstd(keys=["image"], channel_dim=0), 
            
            #transforms.ScaleIntensityd(keys=["image", "low_res_image"], minv=0.0, maxv=1.0),
            #transforms.CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size, roi_image_size)),
            #transforms.CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size, roi_low_res_size)),
            
            Normalization_std(keys=["image"], mean=dict['mean_pixel_img'], std=dict['std_pixel_img']),
            Normalization_std(keys=["low_res_image"], mean=dict['mean_pixel_low_img'], std=dict['std_pixel_low_img']),
            
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
            
            #transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(keys=["low_res_image",'low_res_image','image_vmi_31','image_lumen_36','image_iodine_38'],spatial_size=(low_res_size, low_res_size)),
            transforms.ThresholdIntensityd(keys=["image",'low_res_image','image_vmi_31','image_lumen_36','image_iodine_38'], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["image",'low_res_image','image_vmi_31','image_lumen_36','image_iodine_38'], threshold=0, above=True, cval=0),
            #ApplyTokenizerd(keys=["report"]),
            transforms.ToTensord(keys=["image",'report','low_res_image','image_vmi_31','image_lumen_36','image_iodine_38']) 
        ]
    )
    
    val_transforms = transforms.Compose(
        [   
            #transforms.LoadImaged(keys=["image", "low_res_image"]),
            #transforms.LoadImaged(keys=["image"], reader='PILReader'),
            LoadDICOM(keys=['image','low_res_image','image_vmi_31','image_lumen_36','image_iodine_38']),
            #LoadDICOMmask(keys=['image']),

            transforms.EnsureChannelFirstd(keys=["image", 'low_res_image','image_vmi_31','image_lumen_36','image_iodine_38'], channel_dim=0),
            
            Normalization_std(keys=["image"], mean=dict['mean_pixel_img'], std=dict['std_pixel_img']),
            Normalization_std(keys=["low_res_image"], mean=dict['mean_pixel_low_img'], std=dict['std_pixel_low_img']),
           
            #transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(keys=["low_res_image",'low_res_image','image_vmi_31','image_lumen_36','image_iodine_38'],spatial_size=(low_res_size, low_res_size)),
            transforms.ThresholdIntensityd(keys=["image",'low_res_image','image_vmi_31','image_lumen_36','image_iodine_38'], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["image",'low_res_image','image_vmi_31','image_lumen_36','image_iodine_38'], threshold=0, above=True, cval=0),
            #ApplyTokenizerd(keys=["report"]),
            transforms.ToTensord(keys=["image",'report','low_res_image','image_vmi_31','image_lumen_36','image_iodine_38']) 
        ]
    )
    #out_cache_dir = '/project/outputs/cache/'
    #os.makedirs(out_cache_dir, exist_ok=True)
    #cache_dir = os.path.join(out_cache_dir, "cached_data_aekl_multiclass_mask") 
    #os.makedirs(cache_dir, exist_ok=True)

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
    roi_image_size = 512 # 416
    roi_low_res_size = 358 # 291
    low_res_size =  512 # 208
        
    train_datalist = get_datalist(ids_path=training_ids)
    val_datalist = get_datalist(ids_path=validation_ids)[:1000]

    #img_max_pixel, img_low_max_pixel = get_max_pixel_values(train_datalist) #16254 #15971
    #dict = get_max_min_pixel_values(train_datalist)
    #print(dict['max_pixel_img'])
    #print(dict['min_pixel_img'])
    train_transforms = transforms.Compose(
        [
            #transforms.LoadImaged(keys=["image", "low_res_image"], reader='PILReader'),
            #transforms.LoadImaged(keys=["image"], reader='PILReader'),
            #LoadDICOM(keys=['image']),
            LoadDICOMmask(keys=['image']),

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
            transforms.ToTensord(keys=["image",'low_res_image']) 
        ]
    )
    
    val_transforms = transforms.Compose(
        [
            #transforms.LoadImaged(keys=["image", "low_res_image"], reader='PILReader'),
            #transforms.LoadImaged(keys=["image"], reader='PILReader'),
            #LoadDICOM(keys=['image']),
            LoadDICOMmask(keys=['image']),
            
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
            transforms.ToTensord(keys=["image",'low_res_image']) 
            
        ]
    )

    train_ds = CacheDataset(data=train_datalist, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=True)
    

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