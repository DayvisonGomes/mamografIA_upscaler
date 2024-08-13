import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from monai.transforms import LoadImaged, EnsureChannelFirstd, CenterSpatialCropD, Compose, RandFlipd, RandRotate90d
import pydicom
import tifffile
from tqdm import tqdm

# def creating_csvs(args):
#     """
#     Função que cria os arquivos .tsv em uma pasta passada pelo argumento 
#     output_dir. Esses arquivos contêm os caminhos para cada subset (treino, teste 
#     e validação) das imagens médicas.

#     Ao preencher a lista all_data, é necessário adicionar um dicionário para cada 
#     imagem, permitindo a aplicação de transformações nessas imagens.

#     Args:
#         args (Namespace): Argumento que contém o caminho parao diretório de 
#         saída (--output_dir).
#     """
    
#     path_data = '/project/data/data_tcc' 
#     images_paths = os.listdir(path_data)
#     path_low_res_images = '/project/data/data_tcc_low_res'

#     all_data = []
#     for image_path in images_paths:
#         if '.tif' in image_path:
#             print(image_path)
#             continue
        
#         path_file = os.path.join(path_data, image_path)
#         all_data.append({"image": path_file, "low_res_image": os.path.join(path_low_res_images, image_path)})

#     df = pd.DataFrame(all_data)
#     df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#     train_data_list, test_val_data = train_test_split(df, test_size=0.2, random_state=42)
#     val_data_list, test_data_list = train_test_split(test_val_data, test_size=0.5, random_state=42)

#     train_data_list = pd.concat([train_data_list, test_data_list])
    
#     output_dir = args.output_dir
#     os.makedirs(output_dir, exist_ok=True)

#     train_data_tsv_path = os.path.join(output_dir, "train.tsv")
#     val_data_tsv_path = os.path.join(output_dir, "validation.tsv")
#     test_data_tsv_path = os.path.join(output_dir, "test.tsv")

#     train_data_list.to_csv(train_data_tsv_path, index=False, sep="\t")
#     val_data_list.to_csv(val_data_tsv_path, index=False, sep="\t")
#     test_data_list.to_csv(test_data_tsv_path, index=False, sep="\t")


def generate_data_flip(path_out, path_out_low=None, datalist=None, df=None, eixo=None):
    roi_image_size = 512  
    roi_low_res_size = 358 
    
    path_imgs_list = []  # Lista para armazenar os caminhos das imagens
    path_imgs_low_list = []  # Lista para armazenar os caminhos das imagens de baixa resolução
    indices = []  # Lista para armazenar as classes das imagens
    generated_indice = f'generated_flip_{eixo}'  # Classe para imagens geradas
    classes = []
    
    if eixo == 90:
        
        load_transforms = Compose([
                #LoadImaged(keys=["image", "low_res_image"], reader='PILReader'),
                #EnsureChannelFirstd(keys=["image", "low_res_image"]),
                LoadImaged(keys=["image"], reader='PILReader'),
                EnsureChannelFirstd(keys=["image"]),
                #CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size, roi_image_size)),
                #CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size, roi_low_res_size)),
                #RandRotate90d(keys=["image", "low_res_image"],prob=1.0)
                RandRotate90d(keys=["image"],prob=1.0)

            ])
    else:
        load_transforms = Compose([
                #LoadImaged(keys=["image", "low_res_image"], reader='PILReader'),
                #EnsureChannelFirstd(keys=["image", "low_res_image"]),
                LoadImaged(keys=["image"], reader='PILReader'),
                EnsureChannelFirstd(keys=["image"]),
                #CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size, roi_image_size)),
                #CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size, roi_low_res_size)),
                #RandFlipd(keys=["image", "low_res_image"], spatial_axis=eixo, prob=1.0)
                RandFlipd(keys=["image"], spatial_axis=eixo, prob=1.0)

            ])
        
    for data_dict in tqdm(datalist, desc='Processing images'):
        if 'flipped' in data_dict['image']:
            continue
        loaded_data = load_transforms(data_dict)
        image = loaded_data["image"]
        #low_res_image = loaded_data["low_res_image"]
        #classe = data_dict['classe']
        
        name_img = os.path.splitext(os.path.basename(data_dict['image']))[0]  # Extrair nome do arquivo sem extensão
        #name_img_low = os.path.splitext(os.path.basename(data_dict['low_res_image']))[0]  # Extrair nome do arquivo sem extensão

        path_img = os.path.join(path_out, name_img + f'_flipped_{eixo}.tif')  
        #path_img_low = os.path.join(path_out_low, name_img_low + f'_flipped_{eixo}.tif')
        
        tifffile.imwrite(path_img, image.cpu().numpy())
        #tifffile.imwrite(path_img_low, low_res_image.cpu().numpy())
        
        path_imgs_list.append(path_img)
        #path_imgs_low_list.append(path_img_low)
        indices.append(generated_indice)
        #classes.append(classe)

    # Adicionar as listas ao DataFrame
    df_copy = pd.DataFrame({
        'image': path_imgs_list,
        #'low_res_image': path_imgs_low_list,
        #'indice': indices,
        #'classe': classes
    })
    
    # Concatenar o DataFrame gerado com o DataFrame original
    df_copy = pd.concat([df, df_copy], ignore_index=True)
    
    return df_copy

    
def assign_class(value):
    if 0 <= value <= 124:
        return 'Classe A'
    
    if 250 <= value <= 374:
        return 'Classe B'
    
    if 375 <= value <= 499:
        return 'Classe C'
    
    if 500 <= value <= 623:
        if (value - 500) % 3 == 0:
            return 'Classe A'
        
    if 501 <= value <= 624:
        if (value - 501) % 3 == 0:
            return 'Classe B'
    if 502 <= value <= 622:
        if (value - 502) % 3 == 0:
            return 'Classe C'
    if 675 <= value <= 798:
        if (value - 675) % 3 == 0:
            return 'Classe A'
    if 676 <= value <= 799:
        if (value - 676) % 3 == 0:
            return 'Classe C'
    if 677 <= value <= 797:
        if (value - 677) % 3 == 0:
            return 'Classe C'
    
def get_datalist(ids_path:str):
    """
    Carregamento da tabela dos caminhos para a criação de um vetor com dicionários
    para passar no dataloader específico.

    Args:
        args (str): Caminho do .tsv
    """
    if type(ids_path) is not str:
        df = ids_path
    else:
        df = pd.read_csv(ids_path, sep="\t")

    data_dicts = []
    for index, row in df.iterrows():
#         if '.tif' in row['image']:
#             continue
        
        data_dicts.append(
            {
                "image": str(row["image"]),
                #"low_res_image": str(row['low_res_image']),
                #"classe": str(row['classe'])
            }
        )
    print(f"{len(data_dicts)} imagens.")

    return data_dicts
    
def extrair_indice(path):
    return int(path.split('/')[-1].split('-')[0].split('_')[2])

def creating_csvs(args):
    """
    Função que cria os arquivos .tsv em uma pasta passada pelo argumento
    output_dir. Esses arquivos contêm os caminhos para cada subset (treino, teste
    e validação) das imagens médicas.

    Ao preencher a lista all_data, é necessário adicionar um dicionário para cada
    imagem, permitindo a aplicação de transformações nessas imagens.
    """
    path_data = '/project/data/clinical_zero_padding' 
    #one_path = '/project/data/breast-roi-embed-class-0-512-filter'
    #second_path = '/project/data/breast-roi-embed-class-beta-512-filter'
    images_paths = os.listdir(path_data)

    #images_path_0 = os.listdir(one_path)
    #images_path_rest = os.listdir(second_path)
    
    #path_low_res_images = '/project/data/data_tcc_low_res_crop_tif'

    all_data = []
    # for image_path in images_paths:
    #     path_file = os.path.join(one_path, image_path)
    #     #all_data.append({"image": path_file, "low_res_image": os.path.join(path_low_res_images, image_path)})
    #     all_data.append({"image": path_file})
    indices = [  11,   33,   41,   42,   46,   47,   73,   88,   92,
         95,   98,  106,  109,  125,  133,  198,  208,  215,
        225,  246,  303,  355,  359,  364,  404,  414,  423,
        436,  512,  547,  576,  642,  717,  718,  722,  723,
        733,  734,  738,  745,  754,  761,  838,  993, 1028,
       1029, 1042, 1050, 1097, 1160, 1229, 1250, 1254, 1263,
       1268, 1270] # imagens ruins, para remover
    
    for i, image_path in enumerate(images_paths):
        if i+1 in indices:
            continue
        
        path_file = os.path.join(path_data, image_path)
        #all_data.append({"image": path_file, "low_res_image": os.path.join(path_low_res_images, image_path)})
        all_data.append({"image": path_file})
    
    # for image_path in images_path_rest:
    #     path_file = os.path.join(second_path, image_path)
    #     #all_data.append({"image": path_file, "low_res_image": os.path.join(path_low_res_images, image_path)})
    #     all_data.append({"image": path_file})
        
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # roi_image_size = 512  
    # roi_low_res_size = 358 
    
    # load_transforms = Compose([
    #             LoadImaged(keys=["image", "low_res_image"], reader='PILReader'),
    #             EnsureChannelFirstd(keys=["image", "low_res_image"]),
    #             CenterSpatialCropD(keys=["image"], roi_size=(roi_image_size, roi_image_size)),
    #             CenterSpatialCropD(keys=["low_res_image"], roi_size=(roi_low_res_size, roi_low_res_size)),
    # ])

    # df['indice'] = df['image'].apply(lambda x: extrair_indice(x))
    # df['classe'] = df['indice'].apply(lambda x: assign_class(x))
    # all_data = get_datalist(df)
    # path_out = '/project/data/data_tcc_crop_tif' 
    # path_out_low = '/project/data/data_tcc_low_res_crop_tif'
    # os.makedirs(path_out, exist_ok=True)
    # os.makedirs(path_out_low, exist_ok=True)

    # for data_dict in tqdm(all_data, desc='Processing images'):
    #     loaded_data = load_transforms(data_dict)
    #     image = loaded_data["image"]
    #     low_res_image = loaded_data["low_res_image"]
        
    #     name_img = os.path.splitext(os.path.basename(data_dict['image']))[0]  # Extrair nome do arquivo sem extensão
    #     name_img_low = os.path.splitext(os.path.basename(data_dict['low_res_image']))[0]  # Extrair nome do arquivo sem extensão

    #     path_img = os.path.join(path_out, name_img + f'.tif')  
    #     path_img_low = os.path.join(path_out_low, name_img_low + f'.tif')
        
    #     tifffile.imwrite(path_img, image.cpu().numpy())
    #     tifffile.imwrite(path_img_low, low_res_image.cpu().numpy())
    #df['indice'] = df['image'].apply(lambda x: extrair_indice(x))
    #df['classe'] = df['indice'].apply(lambda x: assign_class(x))
    #min_samples = df.groupby('classe').size().min()
    #df = df.groupby('classe').apply(lambda x: x.sample(min_samples)).reset_index(drop=True)
    
    train_data_list, test_val_data = train_test_split(df, test_size=0.2, random_state=42)
    val_data_list, test_data_list = train_test_split(test_val_data, test_size=0.5, random_state=42)
    
    #train_data_list = pd.concat([train_data_list, test_data_list])
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_data_tsv_path = os.path.join(output_dir, "train.tsv")
    val_data_tsv_path = os.path.join(output_dir, "validation.tsv")
    test_data_tsv_path = os.path.join(output_dir, "test.tsv")

    train_data_list.to_csv(train_data_tsv_path, index=False, sep="\t")
    val_data_list.to_csv(val_data_tsv_path, index=False, sep="\t")
    test_data_list.to_csv(test_data_tsv_path, index=False, sep="\t")

    # GENERATE NEW IMAGES IF YOU NEED
    datalist_train = get_datalist(train_data_tsv_path)
    path_out = '/project/data/clinical_zero_padding_data_generated_train'
    #path_out_low = '/project/data/data_generated_low_train'
    os.makedirs(path_out, exist_ok=True)
    #os.makedirs(path_out_low, exist_ok=True)

    #train_data_list = generate_data_flip(path_out, path_out_low ,datalist_train, train_data_list, 0)
    train_data_list = generate_data_flip(path_out , 'path_out_low' ,datalist_train, train_data_list, 0)

    datalist_train = get_datalist(train_data_list)
    #train_data_list = generate_data_flip(path_out, path_out_low ,datalist_train, train_data_list, 1)
    train_data_list = generate_data_flip(path_out, 'path_out_low' ,datalist_train, train_data_list, 1)

    datalist_train = get_datalist(train_data_list)
    
    #train_data_list = generate_data_flip(path_out, path_out_low ,datalist_train, train_data_list, 90)
    train_data_list = generate_data_flip(path_out, 'path_out_low' ,datalist_train, train_data_list, 90)

    train_data_list = train_data_list.sample(frac=1, random_state=42)
    train_data_list.to_csv(train_data_tsv_path, index=False, sep="\t")  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="Caminho em que será criado os \
                        arquivos .tsv")
    args = parser.parse_args()
    creating_csvs(args)