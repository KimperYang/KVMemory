conda activate kvm
HF_ALLOW_CODE_EVAL="1"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=training_res/sum/sum_5_remove_sftmem/checkpoint-6000,revision=step6000,dtype="float" \
    --tasks  gsm8k_cot_llama,ifeval \
    --confirm_run_unsafe_code \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size 32
    # --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,dtype="float" \
