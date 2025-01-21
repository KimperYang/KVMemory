#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate /dccstor/scllm/envs/torchtune/
cd /dccstor/scllm/KVMemory

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_Instruct_2epoch" --pos 0 --ckpt 1186 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_Instruct_2epoch" --pos 1 --ckpt 1186 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_Instruct_2epoch" --pos 2 --ckpt 1186 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_Instruct_2epoch" --pos 3 --ckpt 1186 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_Instruct_2epoch" --pos 4 --ckpt 1186 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_Instruct_2epoch" --pos 5 --ckpt 1186 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_Instruct_2epoch" --pos 6 --ckpt 1186 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/nq/nq_compress.py --run "compress/compress_qa_50_Instruct_2epoch" --pos 7 --ckpt 1186 &

wait
