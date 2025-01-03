#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

python scripts/evaluation/nq/nq_seq.py --ckpt 10000 --pos 0
python scripts/evaluation/nq/nq_seq.py --ckpt 10000 --pos 1
python scripts/evaluation/nq/nq_seq.py --ckpt 10000 --pos 2
python scripts/evaluation/nq/nq_seq.py --ckpt 10000 --pos 3
python scripts/evaluation/nq/nq_seq.py --ckpt 10000 --pos 4
python scripts/evaluation/nq/nq_seq.py --ckpt 10000 --pos 5
python scripts/evaluation/nq/nq_seq.py --ckpt 10000 --pos 6
python scripts/evaluation/nq/nq_seq.py --ckpt 10000 --pos 7
python scripts/evaluation/nq/nq_seq.py --ckpt 10000 --pos 8
python scripts/evaluation/nq/nq_seq.py --ckpt 10000 --pos 9
python scripts/evaluation/nq/nq_seq.py --ckpt 8000 --pos 0
python scripts/evaluation/nq/nq_seq.py --ckpt 8000 --pos 1
python scripts/evaluation/nq/nq_seq.py --ckpt 8000 --pos 2
python scripts/evaluation/nq/nq_seq.py --ckpt 8000 --pos 3
python scripts/evaluation/nq/nq_seq.py --ckpt 8000 --pos 4
python scripts/evaluation/nq/nq_seq.py --ckpt 8000 --pos 5
python scripts/evaluation/nq/nq_seq.py --ckpt 8000 --pos 6
python scripts/evaluation/nq/nq_seq.py --ckpt 8000 --pos 7
python scripts/evaluation/nq/nq_seq.py --ckpt 8000 --pos 8
python scripts/evaluation/nq/nq_seq.py --ckpt 8000 --pos 9
python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 0
python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 1
python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 2
python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 3
python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 4
python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 5
python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 6
python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 7
python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 8
python scripts/evaluation/nq/nq_seq.py --ckpt 6000 --pos 9