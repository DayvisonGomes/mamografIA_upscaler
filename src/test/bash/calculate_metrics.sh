#!/bin/bash

DOCKER_IMAGE="tcc"
SEED=42
TEST_IDS="/project/outputs/tsv_files/test.tsv"
SAMPLES_DIR='/project/outputs/upsampled_imgs'
NUM_WORKERS=4

docker run -it --ipc=host \
    -v $(pwd):/project/ \
    --gpus all \
    $DOCKER_IMAGE python /project/src/test/calculate_metrics.py \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --samples_dir $SAMPLES_DIR \
    --test_ids $TEST_IDS \
    --num_workers $NUM_WORKERS

