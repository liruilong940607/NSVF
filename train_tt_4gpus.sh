# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA=$1
RES="1080x1920"
VALIDRES="540x960"  # the original size maybe too slow for evaluation
                    # we can optionally half the image size only for validation
ARCH="nsvf_base"
SUFFIX="v1"
DATASET=/home/ICT2000/rli/data/nsvf_data/TanksAndTemple/${DATA}
SAVE=/home/ICT2000/rli/projects/nsvf_ckpt/$DATA
MODEL=$ARCH$SUFFIX
mkdir -p $SAVE/$MODEL

# start training locally
python train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..133" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 1 \
    --pixel-per-view 1024 \
    --valid-chunk-size 128 \
    --no-preload\
    --sampling-on-mask 1.0 --no-sampling-at-reader \
    --valid-view-resolution $VALIDRES \
    --valid-views "133..152" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" \
    --background-stop-gradient \
    --arch $ARCH \
    --initial-boundingbox ${DATASET}/bbox.txt \
    --raymarching-stepsize-ratio 0.125 \
    --discrete-regularization \
    --color-weight 128.0 \
    --alpha-weight 1.0 \
    --optimizer "adam" \
    --adam-betas "(0.9, 0.999)" \
    --lr-scheduler "polynomial_decay" \
    --total-num-update 600000 \
    --lr 0.001 \
    --clip-norm 0.0 \
    --criterion "srn_loss" \
    --num-workers 0 \
    --seed 2 \
    --save-interval-updates 2000 --max-update 600000 \
    --virtual-epoch-steps 20000 --save-interval 1 \
    --half-voxel-size-at  "20000,100000,300000" \
    --reduce-step-size-at "20000,100000,300000" \
    --pruning-every-steps 10000 \
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --tensorboard-logdir ${SAVE}/tensorboard/${MODEL} \
    --save-dir ${SAVE}/${MODEL}
