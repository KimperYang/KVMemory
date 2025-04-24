# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/step8.yaml --main_process_port 25678 sum_attn_trainer5.py

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_prompt --reencode 5 --ckpt 6000 --pos 6 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_prompt --reencode 5 --ckpt 6000 --pos 7 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_prompt --reencode 5 --ckpt 6000 --pos 9 &

# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_prompt --reencode 0 --ckpt 6000 --pos 6 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_prompt --reencode 0 --ckpt 6000 --pos 7 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_prompt --reencode 0 --ckpt 6000 --pos 9 &

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua_hqa.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua_musique.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress/lingua_tqa.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress/lingua_wiki.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 &

CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/compress/lingua_hqa.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/compress/lingua_musique.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/compress/lingua_tqa.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/compress/lingua_wiki.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 &

wait