# conda activate unlearning
HF_ALLOW_CODE_EVAL="1"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --num_machines 1 -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype="float" \
    --tasks gsm8k_cot_llama,ifeval \
    --batch_size 8 \
    --apply_chat_template \
    --fewshot_as_multiturn \

    # --tasks  arc_easy,arc_challenge,hellaswag,winogrande,piqa,sciq \
    # --confirm_run_unsafe_code \

    
# training_res/upper_31_8B/checkpoint-6000
