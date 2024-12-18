#!/bin/bash

source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDA_HOME="$CONDA_PREFIX"

# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME="ib,bond"
# export NCCL_IB_CUDA_SUPPORT=1
# export NCCL_IB_DISABLE=0
# export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_NTHREADS=16

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


export LAUNCHER="accelerate launch \
    --config_file /dccstor/scllm/.cache/accelerate/default_config.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes 32 \
    --num_machines $NNODES \
    "

export SCRIPT="baseline_attn_trainer.py"

export CMD="$LAUNCHER $SCRIPT"

echo ${CMD}

$CMD

# echo ${MASTER_PORT}