import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import pandas as pd    
import json
import datetime
import string
import time
from typing import List
import regex

import argparse

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--run', type=str, required=True, help='Path under training_res')
parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')
parser.add_argument('--pos', type=int, required=True, help='Position value')

args = parser.parse_args()

run_name = args.run
ckpt = args.ckpt
pos = args.pos

if pos in [0, 4, 9]:
    jsonObj = pd.read_json(path_or_buf=f'data/raw/nq/nq-open-10_{pos}.jsonl', lines=True)
else:
    jsonObj = pd.read_json(path_or_buf='data/raw/nq/nq-open-10_0.jsonl', lines=True)

# global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")

# global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)
global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16)


# vocab_size = len(global_tokenizer)
# base_model.resize_token_embeddings(vocab_size)

# peft_config_path = "/mnt/data/jingbo/kv_dump_combine_mix5_5000steps/checkpoint-5000"  # Path to the directory where LoRA weights are stored

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

def inference(input_ids):

    tokenizer = global_tokenizer
    model = global_model

    model.eval()

    with torch.no_grad():

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
            top_p=1.0
        )
    # print(outputs)
    generated_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return generated_sequences

def main():
    global_model.to('cuda')
    # template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n"

    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."

    # total_num = len(jsonObj)
    total_num = 500
    correct_num = 0
    res_list = []
    execution_time = 0
    for i in range(total_num):

        print("Processing sample:", str(i))
        memory_list = []
        doc_list = []

        for k in range(0,10):
            title = jsonObj["ctxs"][i][k]["title"]
            text = jsonObj["ctxs"][i][k]["text"]
            doc_list.append({'title': title, 'text':text})

        if pos not in [0,4,9]:
            ground_truth = doc_list.pop(0)
            doc_list.insert(pos, ground_truth)

        for j in range(0,10):
            title = doc_list[j]["title"]
            text = doc_list[j]["text"]
            memory_list.append(f"Document [{j+1}](Title: {title}) {text}\n")

        memory_list.insert(0, template)

        prompt = "".join(memory_list)
        new_prompt = "\n\nQuestion: " + jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt += new_prompt
        # prompt = template + new_prompt

        prompt_id = global_tokenizer(prompt, return_tensors="pt").input_ids

        # start_time = time.time()
        generated_seq = inference(prompt_id.to(global_model.device))
        # end_time = time.time()
        # if i >= 100:
        #     execution_time += end_time - start_time
        # print(f"Execution time: {execution_time} seconds")
        response = generated_seq[0].split('assistant\n\n')[-1]
        print(response)

        score = best_subspan_em(response, jsonObj["answers"][i])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": jsonObj["question"][i], "response": response, "gold_answer": jsonObj["answers"][i], "Score": score})
        print("Correct progress", correct_num)

    accuracy = correct_num / total_num
    print(accuracy)
    # print(f"Average execution time: {execution_time / 400} seconds")
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    # file_name = f"result/upper/NQ_{run_name}_ckpt{ckpt}_at{pos}_{accuracy}_{time_str}.jsonl"
    file_name = f"result/order/original/NQ_ckpt{ckpt}_at{pos}_{accuracy}_{time_str}.jsonl"
    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
