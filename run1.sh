# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/batch/nq_turbo_nocontext.py --pos 20 --ckpt_path "meta-llama/Llama-3.2-1B-Instruct" --batch_size 8 --attn_type "blocked" --weight 1 --hf True &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/batch/nq_turbo_nocontext.py --pos 20 --ckpt_path "meta-llama/Llama-3.2-3B-Instruct" --batch_size 8 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/batch/nq_turbo_nocontext.py --pos 20 --ckpt_path "meta-llama/Llama-3.1-8B-Instruct" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/batch/nq_turbo.py --pos 0 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/batch/nq_turbo.py --pos 1 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/batch/nq_turbo.py --pos 2 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/batch/nq_turbo.py --pos 3 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/batch/nq_turbo.py --pos 4 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/batch/nq_turbo.py --pos 5 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/batch/nq_turbo.py --pos 6 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/batch/nq_turbo.py --pos 7 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 3 --hf True &
# wait
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/batch/nq_turbo.py --pos 0 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/batch/nq_turbo.py --pos 1 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/batch/nq_turbo.py --pos 2 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/batch/nq_turbo.py --pos 3 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/batch/nq_turbo.py --pos 4 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/batch/nq_turbo.py --pos 5 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/batch/nq_turbo.py --pos 6 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/batch/nq_turbo.py --pos 7 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# wait
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/batch/nq_turbo.py --pos 8 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/batch/nq_turbo.py --pos 9 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/batch/nq_turbo.py --pos 8 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/batch/nq_turbo.py --pos 9 --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/batch/wiki_turbo.py --ckpt_path "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --batch_size 8 --attn_type "blocked" --weight 3 --hf True &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/batch/wiki_turbo.py --ckpt_path "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --batch_size 4 --attn_type "blocked" --weight 8 --hf True &

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/2wiki/wiki_turbo.py --run "/home/azureuser/cloudfiles/code/Users/jingbo.yang/KVMemory/training_res/turbo/turbo_1B/checkpoint-6000" --weight 1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_turbo.py --run "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" --weight 3 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/2wiki/wiki_turbo.py --run "/mnt/tmp/training_res/turbo/turbo_8B/checkpoint-6000" --weight 8 &
wait