#!/bin/bash

DOCKER_IMAGE="tcc"
SEED=42
TRAINING_IDS="/project/outputs/tsv_files/train.tsv"
VALIDATION_IDS="/project/outputs/tsv_files/validation.tsv"
CONFIG_FILE="/project/configs/aekl_configs/aekl_v0.yaml"
BATCH_SIZE=1
N_EPOCHS=30
AUTOENCODER_WARM_UP_N_EPOCHS=10
VAL_INTERVAL=10
NUM_WORKERS=8

docker run -it --ipc=host \
    -v $(pwd):/project/ \
    --gpus all \
    $DOCKER_IMAGE python /project/src/train/train_aekl.py \
    --seed $SEED \
    --training_ids $TRAINING_IDS \
    --validation_ids $VALIDATION_IDS \
    --config_file $CONFIG_FILE \
    --batch_size $BATCH_SIZE \
    --n_epochs $N_EPOCHS \
    --autoencoder_warm_up_n_epochs $AUTOENCODER_WARM_UP_N_EPOCHS \
    --val_interval $VAL_INTERVAL \
    --num_workers $NUM_WORKERS