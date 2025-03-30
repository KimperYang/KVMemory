# conda activate unlearning
HF_ALLOW_CODE_EVAL="1"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m lm_eval --model hf \
    --num_processes 8 \
    --num_machines 1 \
    --model_args pretrained=training_res/new_data/upper_8B/checkpoint-6000,dtype="float" \
    --tasks  arc_easy,arc_challenge,hellaswag,winogrande,piqa,sciq \
    --batch_size 16


    # --confirm_run_unsafe_code \
    # --apply_chat_template \
    # --fewshot_as_multiturn \
    

