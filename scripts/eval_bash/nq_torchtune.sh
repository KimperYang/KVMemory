
#!/bin/bash

# Number of GPUs available
NUM_GPUS=8

# Number of processes to run (for pos=0 to pos=9)
NUM_PROCESSES=10

# Command template
# COMMAND="python scripts/evaluation/nq/nq_torchtune.py --ckpt_path run_logs/decay --batch_size 8"
COMMAND="python scripts/evaluation/nq/nq_torchtune.py --ckpt_path training_res/remove_textinst.pt --batch_size 8 --attn_type blocked"

# Array to keep track of running processes (1 per GPU)
declare -a GPU_PROCESS

# Function to launch a process on a specific GPU
launch_process() {
    local pos=$1
    local gpu=$2
    CUDA_VISIBLE_DEVICES=$gpu $COMMAND --pos $pos &
    GPU_PROCESS[$gpu]=$!
    echo "Launched pos=$pos on GPU=$gpu (PID=${GPU_PROCESS[$gpu]})"
}

# Start launching processes
next_pos=0
while ((next_pos < NUM_PROCESSES || ${#GPU_PROCESS[@]} > 0)); do
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        # Check if the GPU is free or if the process running on it has completed
        if [[ -z ${GPU_PROCESS[$gpu]} ]] || ! kill -0 ${GPU_PROCESS[$gpu]} 2>/dev/null; then
            if ((next_pos < NUM_PROCESSES)); then
                # Launch the next process on this GPU
                launch_process $next_pos $gpu
                next_pos=$((next_pos + 1))
            fi
        fi
    done
    # Sleep for a short time to avoid busy-waiting
    sleep 1
done

echo "All processes have completed."