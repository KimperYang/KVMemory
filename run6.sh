
# python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0 --pos 0
# python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0 --pos 1
# python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0 --pos 2

# python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0 --pos 3
# python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0 --pos 4
# python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0 --pos 5

# python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0 --pos 6
# python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0 --pos 7
# python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0 --pos 8

# python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0 --pos 9

# python scripts/evaluation/tqa/tqa_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0
# python scripts/evaluation/2wiki/wiki_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0
# python scripts/evaluation/musique/musique_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0
# python scripts/evaluation/hqa/hqa_sum.py --ckpt 1122 --run "training_res/sum/sum_0_8B_qa" --reencode 0

CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 0
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 1
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 2
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 3
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 4
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 5
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 6
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 7
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 8
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/nq/nq_upper.py --ckpt 1122 --run "meta-llama/Llama-3.1-8B-Instruct" --pos 9