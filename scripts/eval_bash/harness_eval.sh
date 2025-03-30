# conda activate unlearning
HF_ALLOW_CODE_EVAL="1"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --num_machines 1 -m lm_eval --model hf \
    --model_args pretrained=/mnt/data2/jingbo/sum_5_8B/checkpoint-6000,dtype="float" \
    --tasks  arc_easy,arc_challenge,hellaswag,winogrande,piqa,sciq \
    --batch_size 16


    # --confirm_run_unsafe_code \
    # --apply_chat_template \
    # --fewshot_as_multiturn \
    

