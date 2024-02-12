import pandas as pd
import numpy as np
from monai import transforms
from monai.data import CacheDataset, DataLoader

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
        data_dicts.append(
            {
                "image": str(row["image"]),
                "low_res_image": str(row['low_res_image'])
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
    roi_image_size = 512
    roi_low_res_size = 358
    low_res_size = 256

    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "low_res_image"]),
            transforms.EnsureChannelFirstd(keys=["image", "low_res_image"]),
            transforms.ScaleIntensityd(keys=["image", "low_res_image"], minv=0.0, maxv=1.0),

            transforms.CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size,roi_image_size)),
            transforms.CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size,roi_low_res_size)),

            # transforms.RandFlipd( #
            #     keys=["image"],
            #     spatial_axis=0,
            #     prob=0.5,
            # ),
            # transforms.RandAffined(
            #     keys=["image"],
            #     rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            #     translate_range=[(-1, 1), (-1, 1)],
            #     scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            #     spatial_size=[image_size, image_size], #*
            #     padding_mode="zeros",
            #     prob=0.25,
            # ),
            transforms.Resized(keys=["low_res_image"],
                               spatial_size=(low_res_size, low_res_size)),
            transforms.ToTensord(keys=["image",'low_res_image']),

        ]
    )
    val_transforms = transforms.Compose(
        [
          transforms.LoadImaged(keys=["image", "low_res_image"]),
          transforms.EnsureChannelFirstd(keys=["image", "low_res_image"]),
          transforms.ScaleIntensityd(keys=["image", "low_res_image"], minv=0.0, maxv=1.0),
          transforms.CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size,roi_image_size)),
          transforms.CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size,roi_low_res_size)),
          transforms.Resized(keys=["low_res_image"], spatial_size=(low_res_size, low_res_size)),
          transforms.ToTensord(keys=["image", "low_res_image"]),
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