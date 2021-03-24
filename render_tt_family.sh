#!/bin/bash

DATA="Family"
RES="1080x1920"
VALIDRES="540x960"  # the original size maybe too slow for evaluation
                    # we can optionally half the image size only for validation
ARCH="nsvf_base"
SUFFIX="v2"
DATASET=/home/ICT2000/rli/data/nsvf_data/TanksAndTemple/${DATA}
SAVE=/home/ICT2000/rli/projects/nsvf_ckpt/$DATA
MODEL=$ARCH$SUFFIX
MODEL_PATH=${SAVE}/${MODEL}/checkpoint_last.pt


python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-camera-poses ${DATASET}/pose \
    --render-views "133..152" \
    --model-overrides '{"chunk_size":512,"raymarching_tolerance":0.01}' \
    --render-resolution $RES \
    --render-output ${SAVE}/output \
    --render-output-types "color" \
    --log-format "simple"

