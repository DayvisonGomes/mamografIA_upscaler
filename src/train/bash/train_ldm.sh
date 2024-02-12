#!/bin/bash

DOCKER_IMAGE="tcc"
SEED=42
TRAINING_IDS="/project/outputs/tsv_files/train.tsv"
VALIDATION_IDS="/project/outputs/tsv_files/validation.tsv"
CONFIG_FILE="/project/configs/ldm_configs/ldm_v0.yaml"
BATCH_SIZE=1
N_EPOCHS=120
VAL_INTERVAL=20
NUM_WORKERS=4
CONFIG_FILE_AEKL="/project/configs/aekl_configs/aekl_v0.yaml"
STAGE_1_PATH="/project/outputs/runs/aekl/final_model_aekl.pth"
SCALE_FACTOR=1.0

docker run -it --ipc=host \
    -v $(pwd):/project/ \
    --gpus all \
    $DOCKER_IMAGE python /project/src/train/train_ldm.py \
    --seed $SEED \
    --training_ids $TRAINING_IDS \
    --validation_ids $VALIDATION_IDS \
    --config_file $CONFIG_FILE \
    --stage1_config_file_path $CONFIG_FILE_AEKL \
    --stage1_path $STAGE_1_PATH \
    --scale_factor $SCALE_FACTOR \
    --batch_size $BATCH_SIZE \
    --n_epochs $N_EPOCHS \
    --val_interval $VAL_INTERVAL \
    --num_workers $NUM_WORKERS
