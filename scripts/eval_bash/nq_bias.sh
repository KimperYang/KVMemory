#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

python scripts/evaluation/nq/nq_bias.py --ckpt 2000 --pos 0
python scripts/evaluation/nq/nq_bias.py --ckpt 2000 --pos 4

