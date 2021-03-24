#!/bin/bash

DATA=$1
DATASET=/home/ICT2000/rli/data/nsvf_data/Synthetic_NeRF/$DATA
SAVE=/home/ICT2000/rli/projects/nsvf_ckpt/$DATA

python -u train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..100" --view-resolution "800x800" \
    --max-sentences 1 --view-per-batch 1 --pixel-per-view 2048 \
    --no-preload \
    --sampling-on-mask 1.0 --no-sampling-at-reader \
    --valid-views "100..200" --valid-view-resolution "400x400" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" --background-stop-gradient \
    --arch nsvf_base \
    --initial-boundingbox ${DATASET}/bbox.txt \
    --use-octree \
    --raymarching-stepsize-ratio 0.125 \
    --discrete-regularization \
    --color-weight 128.0 --alpha-weight 1.0 \
    --optimizer "adam" --adam-betas "(0.9, 0.999)" \
    --lr 0.001 --lr-scheduler "polynomial_decay" --total-num-update 600000 \
    --criterion "srn_loss" --clip-norm 0.0 \
    --num-workers 0 \
    --seed 2 \
    --save-interval-updates 2000 --max-update 600000 \
    --virtual-epoch-steps 20000 --save-interval 1 \
    --half-voxel-size-at  "20000,100000,300000" \
    --reduce-step-size-at "20000,100000,300000" \
    --pruning-every-steps 10000 \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --log-format simple --log-interval 1 \
    --save-dir ${SAVE} \
    --tensorboard-logdir ${SAVE}/tensorboard \
    | tee -a $SAVE/train.log

