CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/quant/tqa_sum.py --ckpt 1122 --run "training_res/sum_5_qa" --reencode 5 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/quant/wiki_sum.py --ckpt 1122 --run "training_res/sum_5_qa" --reencode 5 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/quant/musique_sum.py --ckpt 1122 --run "training_res/sum_5_qa" --reencode 5 &
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/quant/hqa_sum.py --ckpt 1122 --run "training_res/sum_5_qa" --reencode 5 &