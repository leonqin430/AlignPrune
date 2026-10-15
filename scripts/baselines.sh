#!/bin/bash
# Baselines: METHOD=full | static | random | infobatch | seta
#   full:      full-data training (200 epochs)
#   static:    static random pruning, a fixed random subset of 1 - RATIO of the data (200 epochs)
#   random:    dynamic random pruning, a new random subset of 1 - RATIO of the data at every epoch (200 epochs)
#   infobatch: InfoBatch (r = 0.5); EPOCHS=196 -> ~30%, 140 -> ~50%, 84 -> ~70% pruning (CIFAR-100N, approx.)
#   seta:      SeTa (r = 0.1, k = 5, alpha = 0.9); EPOCHS=204 -> ~30%, 146 -> ~50%, 87 -> ~70% pruning
# CIFAR-10N: DATASET=cifar10 MAX_LR=5.62 NOISE_TYPE=aggre (or worst). Batch size 128 x 2 GPUs.
METHOD=${METHOD:-infobatch}
RATIO=${RATIO:-0.5}             # pruning ratio of static / random pruning
DATA_PATH=${DATA_PATH:-./data}
DATASET=${DATASET:-cifar100}    # cifar10 | cifar100
NOISE_TYPE=${NOISE_TYPE:-real}  # clean | real (CIFAR-100N) | aggre | worst (CIFAR-10N) | sym | asym
NOISE_RATE=${NOISE_RATE:-0.0}   # rate of the sym / asym noise
MAX_LR=${MAX_LR:-5.2}           # 5.2 for CIFAR-100N, 5.62 for CIFAR-10N

case ${METHOD} in
    full)      PRUNE="--prune_type none";                                   EPOCHS=${EPOCHS:-200} ;;
    static)    PRUNE="--prune_type static --prune_ratio ${RATIO}";          EPOCHS=${EPOCHS:-200} ;;
    random)    PRUNE="--prune_type random --prune_ratio ${RATIO}";          EPOCHS=${EPOCHS:-200} ;;
    infobatch) PRUNE="--prune_type infobatch --prune_ratio 0.5";            EPOCHS=${EPOCHS:-140} ;;
    seta)      PRUNE="--prune_type seta --prune_ratio 0.1 --num_group 5 --window_scale 0.9"; EPOCHS=${EPOCHS:-146} ;;
    *) echo "unknown METHOD=${METHOD}"; exit 1 ;;
esac
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/${METHOD}_${DATASET}_${NOISE_TYPE}_${NOISE_RATE}_ep${EPOCHS}}

torchrun --nproc_per_node=2 main_cifar.py \
    --dataset ${DATASET} --data_path ${DATA_PATH} --noise_type ${NOISE_TYPE} --noise_rate ${NOISE_RATE} \
    ${PRUNE} \
    --epochs ${EPOCHS} --batch_size 128 --optimizer lars --max_lr ${MAX_LR} --amp \
    --output_dir ${OUTPUT_DIR}
