LOG_RANK=${LOG_RANK:-0}
NGPU=${NGPU:-"8"}

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    titan_trainer.py --config_name block_datav3_step10k_bsz64_single_node_selective_ckpt
