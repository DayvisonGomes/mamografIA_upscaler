import os
import shutil
import pandas as pd

#origem = r'D:\Users\dayv\mamografIA_upscaler\outputs\recons_imgs_valid_clinical_franklin_16bits'
origem = r'D:\Users\dayv\baixar'

destino = r"D:\Users\bbruno\Box\Box Sync\Folder_UFPB\Dayvison"

shutil.copy(os.path.join(origem, '2024_07_17-plaq.CT.3DP_HR(Adult).65.10.2024.07.17.22.01.33.343.71822546.dcm'), os.path.join(destino, '2024_07_17-plaq.CT.3DP_HR(Adult).65.10.2024.07.17.22.01.33.343.71822546.dcm'))

# arquivos = os.listdir(origem)[:4]

# for arquivo in arquivos:
#     caminho_origem = os.path.join(origem, arquivo)
#     caminho_destino = os.path.join(destino, arquivo)
#     if os.path.isfile(caminho_origem):
#         shutil.copy(caminho_origem, caminho_destino)
#         print(f"Arquivo {arquivo} copiado com sucesso para o Box Sync!")
