
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/batch/musique_qwen_sum.py --ckpt_path "/mnt/tmp/training_res/sum/sum_5_qwen_2e-5/checkpoint-6000" --batch_size 4 --attn_type "blocked" --reencode_num 5 --hf True &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/batch/hqa_qwen_sum.py --ckpt_path "/mnt/tmp/training_res/sum/sum_5_qwen_2e-5/checkpoint-6000" --batch_size 4 --attn_type "blocked" --reencode_num 5 --hf True &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/batch/hqa_qwen_sum.py --ckpt_path "/mnt/tmp/training_res/sum/sum_5_qwen/checkpoint-6000" --batch_size 4 --attn_type "blocked" --reencode_num 5 --hf True &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/batch/hqa_qwen_blk.py --ckpt_path "/mnt/tmp/training_res/sum/blk_qwen/checkpoint-6000" --batch_size 4 --attn_type "blocked" --hf True &
wait

