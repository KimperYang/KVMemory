#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_bias.py --run "torchtune/bias" --pos 0 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_bias.py --run "torchtune/bias" --pos 1 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_bias.py --run "torchtune/bias" --pos 2 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_bias.py --run "torchtune/bias" --pos 3 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_bias.py --run "torchtune/bias" --pos 4 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_bias.py --run "torchtune/bias" --pos 5 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_bias.py --run "torchtune/bias" --pos 6 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_bias.py --run "torchtune/bias" --pos 7 --ckpt 6000 &

wait
