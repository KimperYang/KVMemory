# KVMemory

## Dependencies
```
pip install transfomers==4.43.1
pip intall datasets==3.6.0
pip install wandb
pip install accelerate
pip install deepspeed
pip install gdown
pip install absl-py
```
Install torch for CUDA12.8
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
##

## DATA
```
python scripts/data_process/fineweb.py --num_samples=10000000 --min_length_for_memory=2048 --validation_size=3000
python scripts/data_process/tulu.py --max_length=4096 --validation_size=2000
python scripts/data_process/daring_anteater.py --max_length=4096 --validation_size=2000
python scripts/data_process/sum.py --max_length=4096 --validation_size=1000
```
For QA dara
```
gdown https://drive.google.com/uc?id=1wjSX2C0OzvmWY18JL76Y0VLj-WiMIA_6
unzip block_qa.zip
mv block_qa_cp dataset_cache/processed/block_qa
```

## Train
Change the DDP config with the steps
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/8gpu_step4.yaml --main_process_port 25678 qwen_sum_trainer.py --bsz 2 --steps 4 --lr 5e-6 --output_dir "/mnt/tmp/training_res/sum/sum_5_qwen_2e-5"
```

## Eval
```
scripts/evaluation/qwen/hqa_sum.py \
    --ckpt_path training_res/sum_5_8B/checkpoint-6000 \
    --batch_size 4 \
    --reencode_num 5 \
    --attn_type "blocked" \
    --hf True
    --output_dir result/qwen_5e-6
```
```
scripts/evaluation/qwen/musique_sum.py \
    --ckpt_path training_res/sum_5_8B/checkpoint-6000 \
    --batch_size 4 \
    --reencode_num 5 \
    --attn_type "blocked" \
    --hf True
    --output_dir result/qwen_5e-6
```
```
scripts/evaluation/qwen/nq_sum.py \
    --ckpt_path training_res/sum_5_8B/checkpoint-6000 \
    --batch_size 4 \
    --reencode_num 5 \
    --attn_type "blocked" \
    --hf True
    --output_dir result/qwen_5e-6
    --pos 0
```
