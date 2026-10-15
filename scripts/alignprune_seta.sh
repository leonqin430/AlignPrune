#!/bin/bash
# AlignPrune on top of SeTa: SeTa's random dropping (r = 0.1), grouping (k = 5) and sliding window (alpha = 0.9)
# operate on the Dynamic Alignment Score (DAS) instead of the loss.
# The overall pruning ratio is set by the number of epochs, relative to 200 epochs of full-data training
# (CIFAR-100N, approximately): EPOCHS=174 -> ~30%, 124 -> ~50%, 75 -> ~70%.
# CIFAR-10N: DATASET=cifar10 MAX_LR=5.62 NOISE_TYPE=aggre (or worst). Batch size 128 x 2 GPUs.
DATA_PATH=${DATA_PATH:-./data}
DATASET=${DATASET:-cifar100}    # cifar10 | cifar100
NOISE_TYPE=${NOISE_TYPE:-real}  # clean | real (CIFAR-100N) | aggre | worst (CIFAR-10N) | sym | asym
NOISE_RATE=${NOISE_RATE:-0.0}   # rate of the sym / asym noise
EPOCHS=${EPOCHS:-124}
MAX_LR=${MAX_LR:-5.2}           # 5.2 for CIFAR-100N, 5.62 for CIFAR-10N
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/alignprune_seta_${DATASET}_${NOISE_TYPE}_${NOISE_RATE}_ep${EPOCHS}}

torchrun --nproc_per_node=2 main_cifar.py \
    --dataset ${DATASET} --data_path ${DATA_PATH} --noise_type ${NOISE_TYPE} --noise_rate ${NOISE_RATE} \
    --prune_type alignprune_seta --prune_ratio 0.1 --num_group 5 --window_scale 0.9 --window_size 25 --score pearson \
    --epochs ${EPOCHS} --batch_size 128 --optimizer lars --max_lr ${MAX_LR} --amp \
    --output_dir ${OUTPUT_DIR}
