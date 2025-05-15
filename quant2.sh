CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --pos 0 --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --pos 1 --reencode 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress/lingua.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --pos 2 --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress/lingua.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --pos 3 --reencode 5 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --pos 4 --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --pos 5 --reencode 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress/lingua.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --pos 6 --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress/lingua.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --pos 7 --reencode 5 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --pos 8 --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --pos 9 --reencode 5 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/compress/lingua_hqa.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compress/lingua_musique.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --reencode 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/compress/lingua_tqa.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/compress/lingua_wiki.py --ckpt 1122 --run "training_res/sum_5_1B_qa" --reencode 5 &