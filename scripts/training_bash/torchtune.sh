#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate /dccstor/scllm/envs/torchtune/
cd /dccstor/scllm/KVMemory

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDA_HOME="$CONDA_PREFIX"

# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME="ib,bond"
# export NCCL_IB_CUDA_SUPPORT=1
# export NCCL_IB_DISABLE=0
# export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_SOCKET_IFNAME=bond1
# export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
# export NCCL_SOCKET_NTHREADS=16

# export NCCL_NET=IB
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_DISABLE=0

# # # Use a temporary file or define hostfile path:
# # HOSTFILE="hostfile"

# # # Clear or create the hostfile:
# # > $HOSTFILE

# # echo "${LSB_MCPU_HOSTS}" | sed 's/[[:space:]]*$//' | \
# # tr ' ' '\n' | paste - - | awk '{print $1 ".pok.ibm.com slots=8"}' > hostfile

startTime=$(date +%s) #mark the start of job
MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
MASTER_PORT=28442 #5${LSB_JOBID: -5:-1}
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
# NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOSTNAME | cut -d':' -f1)-1))
BASE_HOSTNAME=$(echo $HOSTNAME | cut -d'.' -f1)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $BASE_HOSTNAME | cut -d':' -f1)-1))
JOB_ID=${LSB_JOBID}


LOG_RANK=${LOG_RANK:-0}
NGPU=${NGPU:-"8"}
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LAUNCHER="torchrun \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --rdzv_backend c10d \
    --rdzv_endpoint=${MASTER_ADDR}:29500 \
    --local-ranks-filter ${LOG_RANK} \
    --role rank --tee 3 \
    titan_trainer.py --config_name block_datav1_step10k_bsz128_2_node_full_ckpt \
    "

export CMD="$LAUNCHER"

echo ${CMD}

$CMD



