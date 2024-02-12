import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def creating_csvs(args):
    """
    Função que cria os arquivos .tsv em uma pasta passada pelo argumento 
    output_dir. Esses arquivos contêm os caminhos para cada subset (treino, teste 
    e validação) das imagens médicas.

    Ao preencher a lista all_data, é necessário adicionar um dicionário para cada 
    imagem, permitindo a aplicação de transformações nessas imagens.

    Args:
        args (Namespace): Argumento que contém o caminho parao diretório de 
        saída (--output_dir).
    """
    
    path_data = '/project/data/data_tcc' 
    images_paths = os.listdir(path_data)
    path_low_res_images = '/project/data/data_tcc_low_res'

    all_data = []
    for image_path in images_paths:
        path_file = os.path.join(path_data, image_path)
        all_data.append({"image": path_file, "low_res_image": os.path.join(path_low_res_images, image_path)})

    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_data_list, test_val_data = train_test_split(df, test_size=0.2, random_state=42)
    val_data_list, test_data_list = train_test_split(test_val_data, test_size=0.5, random_state=42)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_data_tsv_path = os.path.join(output_dir, "train.tsv")
    val_data_tsv_path = os.path.join(output_dir, "validation.tsv")
    test_data_tsv_path = os.path.join(output_dir, "test.tsv")

    train_data_list.to_csv(train_data_tsv_path, index=False, sep="\t")
    val_data_list.to_csv(val_data_tsv_path, index=False, sep="\t")
    test_data_list.to_csv(test_data_tsv_path, index=False, sep="\t")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="Caminho em que será criado os \
                        arquivos .tsv")
    args = parser.parse_args()
    creating_csvs(args)
