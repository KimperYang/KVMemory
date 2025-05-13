startTime=$(date +%s)

output_dir=$1
run_name=$2

export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file config/4gpu_step4.yaml --main_process_port 25678 sum_qa_trainer.py --reencode 5 --weight 3
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/4gpu_step4.yaml --main_process_port 25678 sum_qa_trainer.py --reencode 1 --weight 3
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/4gpu_step4.yaml --main_process_port 25678 sum_qa_trainer.py --reencode 0 --weight 3
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/4gpu_step4.yaml --main_process_port 25678 sum_qa_trainer.py --reencode 1 --weight 1
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/4gpu_step4.yaml --main_process_port 25678 sum_qa_trainer.py --reencode 0 --weight 1




# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 --pos 2 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 --pos 3 &

# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 --pos 2 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 --pos 3 &
# wait
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 --pos 4 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 --pos 5 &

# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 --pos 4 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 --pos 5 &
# wait
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 --pos 6 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 --pos 7 &

# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 --pos 6 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 --pos 7 &
# wait
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 --pos 8 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 --pos 9 &

# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 --pos 8 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 --pos 9 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/compress/lingua.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 --pos 9 &

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua_hqa.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua_musique.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress/lingua_tqa.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress/lingua_wiki.py --run training_res/sum/sum_5_3B --reencode 5 --ckpt 6000 &

# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/compress/lingua_hqa.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/compress/lingua_musique.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/compress/lingua_tqa.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/compress/lingua_wiki.py --run training_res/sum/sum_0_3B --reencode 0 --ckpt 6000 &

# wait