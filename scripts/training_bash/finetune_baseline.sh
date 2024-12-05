#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

accelerate launch --config_file ../.cache/huggingface/accelerate/default_config.yaml --main_process_port 25678 finetune_baseline.py