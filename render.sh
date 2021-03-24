#!/bin/bash

DATA=$1
DATASET=/home/ICT2000/rli/data/nsvf_data/Synthetic_NeRF/$DATA
SAVE=/home/ICT2000/rli/projects/nsvf_ckpt/$DATA
MODEL_PATH=${SAVE}/checkpoint_last.pt

python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-camera-poses ${DATASET}/pose \
    --render-views "200..400" \
    --model-overrides '{"chunk_size":512,"raymarching_tolerance":0.01}' \
    --render-resolution "800x800" \
    --render-output ${SAVE}/output \
    --render-output-types "color" \
    --log-format "simple"

