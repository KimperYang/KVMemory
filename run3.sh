CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/tqa/tqa_turbo.py  --run "/mnt/tmp/training_res/turbo/turbo_3B/checkpoint-6000" &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/tqa/tqa_turbo.py  --run "training_res/sum_1_31_8B" &

wait