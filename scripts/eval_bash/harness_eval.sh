conda activate kvm
HF_ALLOW_CODE_EVAL="1"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=training_res/new_data/upper_3B/checkpoint-6000,revision=step6000,dtype="float" \
    --tasks  mmlu,arc_easy,arc_challenge,hellaswag,winogrande,piqa,lambada_openai,sciq \
    --batch_size 32
    # --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,dtype="float" \
