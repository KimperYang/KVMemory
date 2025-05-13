CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file config/4gpu_step4.yaml --main_process_port 25679 upper_qa_trainer.py --weight 8
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file config/4gpu_step4.yaml --main_process_port 25679 upper_qa_trainer.py --weight 3
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file config/4gpu_step4.yaml --main_process_port 25679 upper_qa_trainer.py --weight 1
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file config/4gpu_step4.yaml --main_process_port 25679 blk_qa_trainer.py --weight 3
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file config/4gpu_step4.yaml --main_process_port 25679 sum_qa_trainer.py --reencode 0 --weight 8