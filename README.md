# KVMemory

## Run
```
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 finetune.py 
```
```
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --config_file /mnt/data2/jingbo/.cache/huggingface/accelerate/default_config.yaml --main_process_port 25678 finetune.py
```