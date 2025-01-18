python scripts/evaluation/nq/nq_torchtune.py \
    --ckpt_path model_cache/checkpoint.pt \
    --pos 0

CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_torchtune.py \
    --ckpt_path run_logs/decay \
    --pos 0 \
    --batch_size 8


CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_torchtune_v2.py \
    --ckpt_path run_logs/decay \
    --pos 5 \
    --batch_size 1
