source /u/shiyuucsb/.bashrc
conda activate kvm
cd /dccstor/scllm/KVMemory

head_node_id=${}

COMMAND="accelerate launch main_process_id=${head_node_id}"

${COMMAND}


