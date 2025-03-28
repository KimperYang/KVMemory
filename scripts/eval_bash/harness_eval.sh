conda activate unlearning
HF_ALLOW_CODE_EVAL="1"

CUDA_VISIBLE_DEVICES=4,5 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=training_res/sum/sum_5_only_QA/checkpoint-624,revision=step624,dtype="float" \
    --tasks  arc_easy,arc_challenge,hellaswag,winogrande,piqa,sciq \
    --batch_size 16


    # --confirm_run_unsafe_code \
    # --apply_chat_template \
    # --fewshot_as_multiturn \
    

