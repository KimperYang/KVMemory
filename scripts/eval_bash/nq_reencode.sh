#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

python scripts/evaluation/nq/nq_reencode.py --ckpt 8000 --pos 4 --reencode 10
python scripts/evaluation/nq/nq_reencode.py --ckpt 8000 --pos 5 --reencode 10
python scripts/evaluation/nq/nq_reencode.py --ckpt 8000 --pos 6 --reencode 10
python scripts/evaluation/nq/nq_reencode.py --ckpt 8000 --pos 7 --reencode 10
python scripts/evaluation/nq/nq_reencode.py --ckpt 8000 --pos 8 --reencode 10
python scripts/evaluation/nq/nq_reencode.py --ckpt 8000 --pos 9 --reencode 10