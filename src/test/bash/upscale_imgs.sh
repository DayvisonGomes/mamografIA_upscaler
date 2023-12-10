#!/bin/bash

DOCKER_IMAGE="tcc"
SEED=42
TEST_IDS="/project/outputs/tsv_files/test.tsv"
CONFIG_FILE="/project/configs/aekl_configs/aekl_v0.yaml"
OUTPUT_DIR="/project/outputs/upscale_test_set"
CONFIG_FILE_AEKL="/project/configs/aekl_configs/aekl_v0.yaml"
STAGE_1_PATH="/project/outputs/runs/aekl/final_model_aekl.pth"
DIFFUSION_PATH="/project/outputs/runs/ldm/final_model_ldm.pth"
CONFIG_FILE_LDM="/project/configs/ldm_configs/ldm_v0.yaml"
DOWNSAPLED_DIR = '/project/outputs/downsapled_imgs'
START_INDEX=0
STOP_INDEX=5
X_SIZE=64
Y_SIZE=64
SCALE_FACTOR=0.97
INFERENCE_STEPS=1000
NOISE_LEVEL=1

docker run -it --ipc=host \
    -v $(pwd):/project/ \
    --gpus all \
    $DOCKER_IMAGE python /project/src/test/upscale_imgs.py \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --stage1_config_file_path $CONFIG_FILE_AEKL \
    --stage1_path $STAGE_1_PATH \
    --config_file $CONFIG_FILE \
    --diffusion_path $DIFFUSION_PATH \
    --diffusion_config_file_path $CONFIG_FILE_LDM \
    --reference_path $REFERENCE_PATH \
    --start_index $START_INDEX \
    --stop_index $STOP_INDEX \
    --x_size $X_SIZE \
    --y_size $Y_SIZE \
    --scale_factor $SCALE_FACTOR \
    --num_inference_steps $INFERENCE_STEPS \
    --noise_level $NOISE_LEVEL \
    --test_ids $TEST_IDS \
    --downsampled_dir $DOWNSAPLED_DIR
