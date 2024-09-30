# KVMemory

## Run
```
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 finetune.py 
```
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file /mnt/data2/jingbo/.cache/huggingface/accelerate/default_config.yaml --main_process_port 25678 finetune.py
```

## Ifeval
```
python3 -m instruction_following_eval.evaluation_main \
  --input_data=./instruction_following_eval/data/input_data.jsonl \
  --input_response_data=./instruction_following_eval/data/input_response_data_gpt4_20231107_145030.jsonl \
  --output_dir=./instruction_following_eval/data/
```

python3 -m instruction_following_eval.evaluation_main \
  --input_data=/home/jingbo/KVMemory/data/raw/ifeval/input_data.jsonl \
  --input_response_data=/home/jingbo/KVMemory/result/ifeval_combinemodel_20240925-235953.jsonl \
  --output_dir=/home/jingbo/KVMemory/result/ifeval_combine