#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

python scripts/evaluation/2wiki/wiki_bias.py --ckpt 10000
python scripts/evaluation/2wiki/wiki_bias.py --ckpt 6000
python scripts/evaluation/2wiki/wiki_bias.py --ckpt 8000
python scripts/evaluation/2wiki/wiki_bias.py --ckpt 4000
python scripts/evaluation/2wiki/wiki_bias.py --ckpt 2000