#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate /dccstor/scllm/envs/torchtune/
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_20" --pos 0   --ckpt 6000 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50" --pos 1   --ckpt 71 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50" --pos 2   --ckpt 71 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50" --pos 3   --ckpt 71 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50" --pos 4   --ckpt 71 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50" --pos 5   --ckpt 71 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50" --pos 6   --ckpt 71 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50" --pos 7   --ckpt 71 &

wait
