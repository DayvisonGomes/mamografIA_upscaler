import os
import shutil

origem = r'D:\Users\dayv\mamografIA_upscaler\outputs\recons_imgs_valid_clinical_franklin_16bits'
destino = r"D:\Users\bbruno\Box\Box Sync\Folder_UFPB\Dayvison"

arquivos = os.listdir(origem)[:4]

for arquivo in arquivos:
    caminho_origem = os.path.join(origem, arquivo)
    caminho_destino = os.path.join(destino, arquivo)
    if os.path.isfile(caminho_origem):
        shutil.copy(caminho_origem, caminho_destino)
        print(f"Arquivo {arquivo} copiado com sucesso para o Box Sync!")
