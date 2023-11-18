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
    image_size = 732 #*
    low_res_size = 512
    
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
                spatial_size=[image_size, image_size], #*
                padding_mode="zeros",
                prob=0.5,
            ),
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(keys=["low_res_image"], 
                               spatial_size=(low_res_size, low_res_size)),
        ]
    )
    val_transforms = transforms.Compose(
        [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
        transforms.Resized(keys=["low_res_image"], spatial_size=(low_res_size, low_res_size)),
        ]
    )
    
    train_datalist = get_datalist(ids_path=training_ids)
    train_ds = CacheDataset(data=train_datalist, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, persistent_workers=True)
    
    # check_data = first(train_loader)
    # for i in range(3):
    #     plt.imsave(f'/project/outputs/img{i}.png', check_data["image"][i, 0, :, :], cmap="gray")

    val_datalist = get_datalist(ids_path=validation_ids)
    val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers)

    return train_loader, val_loader