import os
import pandas as pd
import numpy as np
from monai import transforms
import cv2
import matplotlib.pyplot as plt
import argparse

class ClassificarBrilho(transforms.Transform):
    def __init__(self, limiar_brilho=0.3):
        self.limiar_brilho = limiar_brilho

    def __call__(self, img):
        brilho_medio = np.mean(img)
        classificacao = "Densa" if brilho_medio > self.limiar_brilho else "Menos Densa"
        return classificacao

def classificar_imagem_com_transformacoes(caminho_imagem, img_transforms, limiar_brilho=0.3):
    img_transformed = img_transforms({"image": caminho_imagem})["image"]
    img_transform = ClassificarBrilho(limiar_brilho=limiar_brilho)
    classificacao = img_transform(img_transformed)

    return classificacao

def percorrer_tsv_e_classificar(tsv_path, img_transforms, limiar_brilho=0.3):
    df = pd.read_csv(tsv_path, sep="\t")

    classificacoes = []

    for index, row in df.iterrows():
        caminho_imagem = str(row["image"])
        classificacao = classificar_imagem_com_transformacoes(caminho_imagem, img_transforms, limiar_brilho=limiar_brilho)
        classificacoes.append({"image": caminho_imagem, "classificacao": classificacao})

    return classificacoes

def chage_tables_ids(args):
    input_dir = args.input_dir
    train_data_tsv_path = os.path.join(input_dir, "train.tsv")
    val_data_tsv_path = os.path.join(input_dir, "validation.tsv")
    test_data_tsv_path = os.path.join(input_dir, "test.tsv")

    img_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        transforms.Resized(keys=["image"], spatial_size=(256, 256)),
    ])

    classificacoes_train = percorrer_tsv_e_classificar(train_data_tsv_path, img_transforms)
    classificacoes_val = percorrer_tsv_e_classificar(val_data_tsv_path, img_transforms)
    classificacoes_test = percorrer_tsv_e_classificar(test_data_tsv_path, img_transforms)

    pd.DataFrame(classificacoes_train).to_csv(os.path.join(input_dir, "classificacoes_train.csv"), index=False)
    pd.DataFrame(classificacoes_val).to_csv(os.path.join(input_dir, "classificacoes_val.csv"), index=False)
    pd.DataFrame(classificacoes_test).to_csv(os.path.join(input_dir, "classificacoes_test.csv"), index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Caminho em que os arquivos .tsv estão")
    args = parser.parse_args()
    chage_tables_ids(args)
