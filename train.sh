CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/step4.yaml --main_process_port 25678 baseline_attn_trainer.py
rm -r training_res/new_data/upper_8B/checkpoint_1000/global_step1000
rm -r training_res/new_data/upper_8B/checkpoint_2000/global_step2000
rm -r training_res/new_data/upper_8B/checkpoint_3000/global_step3000
rm -r training_res/new_data/upper_8B/checkpoint_4000/global_step4000
rm -r training_res/new_data/upper_8B/checkpoint_5000/global_step5000
rm -r training_res/new_data/upper_8B/checkpoint_6000/global_step6000