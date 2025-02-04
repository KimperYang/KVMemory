conda activate unlearning

CUDA_VISIBLE_DEVICES=1,2,5,6 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=training_res/sum/sum_5_prompt/checkpoint-6000,revision=step6000,dtype="float" \
    --tasks  mmlu \
    --batch_size 16

    # --apply_chat_template \
    # --fewshot_as_multiturn \
    # --model_args pretrained=training_res/new_data/upper_prompt/checkpoint-6000,revision=step6000,dtype="float" \
    # --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,dtype="float" \
