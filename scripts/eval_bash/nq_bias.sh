#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate /dccstor/scllm/envs/torchtune/
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_4nodes" --pos 0   --ckpt 149 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_4nodes" --pos 1   --ckpt 149 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_4nodes" --pos 2   --ckpt 149 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_4nodes" --pos 3   --ckpt 149 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_4nodes" --pos 4   --ckpt 149 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_4nodes" --pos 5   --ckpt 149 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_4nodes" --pos 6   --ckpt 149 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_4nodes" --pos 7   --ckpt 149 &

wait
