#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory


export LAUNCHER="accelerate launch \
    --config_file /dccstor/scllm/.cache/accelerate/default_config.yaml \
    --num_processes 8 \
    --num_machines 1 \
    "

export SCRIPT="baseline_attn_trainer.py"

export CMD="$LAUNCHER $SCRIPT"

echo ${CMD}

$CMD
