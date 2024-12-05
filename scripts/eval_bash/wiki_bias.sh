#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

python scripts/evaluation/2wiki/wiki_bias.py