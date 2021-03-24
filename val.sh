#!/bin/bash

DATA=$1
DATASET=/home/ICT2000/rli/data/nsvf_data/Synthetic_NeRF/$DATA
SAVE=/home/ICT2000/rli/projects/nsvf_ckpt/$DATA
MODEL_PATH=${SAVE}/checkpoint_last.pt

python validate.py ${DATASET} \
    --user-dir fairnr \
    --valid-views "200..400" \
    --valid-view-resolution "800x800" \
    --no-preload \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL_PATH} \
    --model-overrides '{"chunk_size":512,"raymarching_tolerance":0.01,"tensorboard_logdir":"","eval_lpips":True}' \


