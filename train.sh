#!/bin/zsh

MAIN_WEIGHTS_FILE="output/exlpose_kpt/hrnet_main/PT_stage_main/final_state0.pth.tar"
COMP_WEIGHTS_FILE="output/exlpose_kpt/hrnet_comp/PT_stage_comp/final_state0.pth.tar"

python tools/train_stage1_PT.py --cfg experiments/exlpose/PT_stage_main.yaml

python tools/train_stage1_PT.py --cfg experiments/exlpose/PT_stage_main.yaml \
    TRAIN.STAGE PT_LL MODEL.PRETRAINED_MAIN ${MAIN_WEIGHTS_FILE}

python tools/train_stage1_PT.py --cfg experiments/exlpose/PT_stage_comp.yaml

python tools/train_stage1_PT.py --cfg experiments/exlpose/PT_stage_comp.yaml \
    TRAIN.STAGE PT_LL MODEL.PRETRAINED_COMP ${COMP_WEIGHTS_FILE}