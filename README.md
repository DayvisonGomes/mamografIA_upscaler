# mamografIA_upscaler

Repositório contendo todos os códigos para a realização da super-resolução em mamografias, utilizando o autoencoder kl e o modelo de difusão da Unet.

Esquema de pastas e arquivos:

```bash
.
├── configs
│   ├── aekl_configs
│   │   └── aekl_v0.yaml
│   ├── ldm_configs
│   │   └── ldm_v0.yaml
├── src
│   ├── preprocessing_data
│   │   ├── create_reference_img.py
│   │   └── process_medical_images.py
│   ├── test
│   │   ├── bash
│   │   │   └── upscale_test_set.sh
│   │   └── upscale_test_set.py
│   ├── train
│   │   ├── bash
│   │   │   ├── train_aekl.sh
│   │   │   └── train_ldm.sh
│   │   ├── train_aekl.py
│   │   ├── train_ldm.py
│   │   ├── util_training.py
│   │   └── util_transformations.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
└── README.md
```
Explicação dos arquivos pela ordem do esquema de pastas e arquivos:

A estrutura do projeto é organizada da seguinte forma:

## Configs

### `aekl_configs`
- Contém configurações relacionadas ao modelo Autoencoder KL.

    - **aekl_v0.yaml**: Arquivo de configuração contendo os hiperparâmetros para a versão 0 do modelo AEKL.

### `ldm_configs`
- Contém configurações relacionadas ao modelo de difusão.

    - **ldm_v0.yaml**: Arquivo de configuração contendo os hiperparâmetros para a versão 0 do modelo LDM.

## SRC

### `preprocessing_data`

- **`create_reference_img.py`**: Script para criar uma imagem de referência que é necessária para o upscale das imagens.

- **`process_medical_images.py`**: Script para processar as imagens médicas, no caso mamografias.

### `test`

#### Bash
- **`upscale_test_set.sh`**: Script bash para rodar o processo de upscale nas imagens do subset de teste.

#### Python
- **`upscale_test_set.py`**: Script Python correspondente ao script bash.

### `train`

#### Bash
- **`train_aekl.sh`**: Script bash para treinar o modelo AEKL.

- **`train_ldm.sh`**: Script bash para treinar o modelo de difusão.

#### Python
- **`train_aekl.py`**: Script Python para treinar o modelo AEKL.

- **`train_ldm.py`**: Script Python para treinar o modelo de difusão.

- **`util_training.py`**: Utilitário com funções relacionadas ao treinamento.

- **`util_transformations.py`**: Utilitário com funções relacionadas às transformações nas imagens, que são necessárias para o treinamento dos modelos.

## Arquivos de Configuração

- **`.dockerignore`**: Arquivo para especificar os arquivos a serem ignorados durante a construção da imagem Docker.

- **`.gitignore`**: Arquivo para especificar os arquivos a serem ignorados pelo Git.

- **`Dockerfile`**: Arquivo para a construção da imagem Docker.

- **`LICENSE`**: Licença do projeto.

## Primeiros passos

Primeiro é preciso ter o docker instalado na sua máquina, após isso é preciso buildar a sua imagem com o seguinte comando:

```bash
docker build -t <image-name> .
```
Isto é, se você estiver no diretório do DockerFile. Ou se preferir, rode o script "create_image_docker.sh" com:

```bash
chmod +x create_image_docker.sh
sh -xe create_image_docker.sh
```

## Pré-processamento

Após a criação da imagem docker, é preciso a criação de uma pasta chamada "data" onde irá conter todos os seus dados para que ao executar o .py "./src/preprocessing_data/process_medical_images.py", não de problemas.

```bash
docker run -it 
        --ipc=host 
        -v ./:/project/ 
        <image-name> 
        python /project/src/preprocessing_data/process_medical_images.py 
            --output_dir project/outputs/tsv_files
```

Esse arquivo é responsável por criar três arquivos .tsv onde estaram os paths de cada imagem, um para cada subset, para editar as quantidades, sugir mudar no próprio arquivo (att futura). O output_dir é onde estará tais arquivos.

## Como treinar o modelo autoencoder

Após a criação dos arquivos .tsv, já é possível o treinamento do autoencoder com o seguinte comando:

```bash
chmod +x ./src/train/bash/train_aekl.sh
sh -xe ./src/train/bash/train_aekl.sh
```

## Como treinar o modelo de difusão latente

Após o treinamento do autoencoder, rode o seguinte comando para treinar o modelo de difusão

```bash
chmod +x ./src/train/bash/train_ldm.sh
sh -xe ./src/train/bash/train_ldm.sh
```

## Como utilizar os modelos para realizar upscale

### Subset de test

Para realizar o upscale em algumas imagens do subset de teste, rode:

```bash
chmod +x ./src/test/bash/upscale_test_set.sh
sh -xe ./src/train/bash/upscale_test_set.sh
```

Para mudar os parâmetros, vá até os arquivos .sh para alterá-los.

### Qualquer imagem

(terminar)

## Dicas e Dificuldades


