
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 --pos 3 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 --pos 4 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 --pos 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 --pos 6 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 --pos 7 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 --pos 8 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/nq/nq_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 --pos 9 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/tqa/tqa_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/2wiki/wiki_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/musique/musique_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/hqa/hqa_sum.py --ckpt 1122 --run "training_res/qa/1B/sum_0_1B_qa" --reencode 0 &