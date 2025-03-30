import argparse
import datetime
import json
import string
from typing import List

import pandas as pd
import regex
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

# vocab_size = len(global_tokenizer)
# base_model.resize_token_embeddings(vocab_size)

# peft_config_path = "/mnt/data/jingbo/kv_dump_combine_mix5_5000steps_5e-6_full/checkpoint-5000"  # Path to the directory where LoRA weights are stored

# global_model = PeftModel.from_pretrained(base_model, peft_config_path)

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def main():

    parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
    parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')
    parser.add_argument('--run', type=str, required=True, help='Checkpoint number')

    args = parser.parse_args()

    ckpt = args.ckpt
    run_name = args.run

    data_list=load_dataset("dgslibisey/MuSiQue", split='validation')

    if "meta" in run_name:
        global_tokenizer = AutoTokenizer.from_pretrained(run_name)
        global_model = AutoModelForCausalLM.from_pretrained(run_name, torch_dtype=torch.bfloat16)
    else:
        global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")
        global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)

    global_model.to('cuda')

    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"

    # total_num = len(jsonObj)
    total_num = len(data_list)
    correct_num = 0
    res_list = []

    for i in range(total_num):

        print("Processing sample:", str(i))
        memory_list = [template]

        for j in range(len(data_list[i]['paragraphs'])):
            title = data_list[i]['paragraphs'][j]['title']
            text = data_list[i]['paragraphs'][j]['paragraph_text']
            memory_list.append(f"Document [{j+1}](Title: {title}) {text}\n")

        id_list = []

        for st in memory_list:

            tem_id = global_tokenizer(st, return_tensors="pt", add_special_tokens=False).input_ids

            id_list.append(tem_id)

        new_prompt = data_list[i]['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt_id = global_tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(global_model.device)

        # id_list.append(prompt_id)

        cache_id = torch.cat(id_list, dim=1).to(global_model.device)

        global_model.eval()

        generate_id = torch.cat([cache_id, prompt_id], dim = 1)

        # print(cache_id.size(1))
        with torch.no_grad():
            outputs = global_model(input_ids = cache_id)
            past_key_values = outputs.past_key_values

            outputs = global_model.generate(
                input_ids=generate_id,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,
                top_p=1.0,
                past_key_values=past_key_values,
                use_cache=True
            )
        # print(outputs)
        generated_seq = global_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
        response = generated_seq[0].split('assistant\n\n')[-1]
        # print(response)
        print("response:", response)

        score = best_subspan_em(response, [data_list[i]['answer']])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": data_list[i]['question'], "response": response, "gold_answer": [data_list[i]['answer']], "Score": score})
        print("Correct progress", correct_num)
        
    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    if "meta" in run_name:
        file_name = f"result/new_data/original_8B/musique_ckpt{ckpt}_{accuracy}_{time_str}.jsonl"
    else:
        file_name = f"result/{run_name}/musique_{accuracy}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
