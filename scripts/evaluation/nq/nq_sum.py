import argparse
import datetime
import json
import os
import string
from typing import List

import pandas as pd
import regex
import torch
from safetensors import safe_open
from torchtune.models.convert_weights import tune_to_hf
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from src.data.attention import construct_biased_attention_matrix

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint number')
parser.add_argument('--pos', type=int, required=True, help='Position value')
parser.add_argument('--reencode', type=int, required=True, help='Reencode num')

args = parser.parse_args()

def load_model_weights(ckpt_path: str):
    safe_tensor_file = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(safe_tensor_file):
        state_dict = {}
        with safe_open(safe_tensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        # state_dict["output.weight"] = state_dict["tok_embeddings.weight"]
        return state_dict

    state_dict = torch.load(ckpt_path, weights_only=False)

    state_dict = state_dict["model"]
    state_dict["output.weight"] = state_dict["tok_embeddings.weight"]

    converted_state_dict = tune_to_hf(
        state_dict=state_dict,
        num_heads=32,
        num_kv_heads=8,
        dim=2048,
    )
    return converted_state_dict


ckpt = args.ckpt
pos = args.pos
reencode_num = args.reencode

if pos in [0, 4, 9]:
    jsonObj = pd.read_json(path_or_buf=f'data/raw/nq/nq-open-10_{pos}.jsonl', lines=True)
else:
    jsonObj = pd.read_json(path_or_buf='data/raw/nq/nq-open-10_0.jsonl', lines=True)

device = torch.device("cuda")
global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
global_tokenizer.pad_token_id = 128004
global_tokenizer.pad_token = "<|finetune_right_pad_id|>"

global_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    # torch_dtype=torch.float32,
)
# model.load_state_dict(state_dict, strict=True)

if args.ckpt is None:
    print("Will NOT load fine-tuned models!")
else:
    state_dict = load_model_weights(args.ckpt)
    global_model.load_state_dict(state_dict, strict=False)
global_model = global_model.to(device)
global_model.eval()

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
    global_model.to('cuda')

    special_token_start=128011
    mem_start=128254
    mem_end=128255

    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>"

    # total_num = len(jsonObj)
    total_num = 500
    correct_num = 0
    res_list = []

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

        # memory_list.insert(0, template)
        sys_id = global_tokenizer(template, add_special_tokens=False).input_ids
        sys_id = sys_id + [mem_start]

        biased_index = []
        concat_id = []

        idx = len(sys_id)

        for j in range(len(memory_list)):

            tem_id = global_tokenizer(memory_list[j], add_special_tokens=False).input_ids

            for sub_idx in range(reencode_num):
                tem_id = tem_id + [special_token_start + reencode_num * j + sub_idx]

            biased_index.append([idx, idx + len(tem_id) - reencode_num])

            concat_id += tem_id

            idx = idx + len(tem_id)

        concat_id = sys_id + concat_id + [mem_end]
        concat_id = torch.tensor([concat_id], device=global_model.device)
        attention_matrix = construct_biased_attention_matrix(concat_id.size(1), biased_index, concat_id.size(1), global_model.device).unsqueeze(0).unsqueeze(0)

        new_prompt = "<|start_header_id|>user<|end_header_id|>\n\nWrite a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Question: " + jsonObj["question"][i] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt_id = global_tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(global_model.device)

        generate_id = torch.cat([concat_id, prompt_id], dim = 1)

        global_model.eval()
        with torch.no_grad():
            outputs = global_model(input_ids = concat_id, attention_mask = attention_matrix)
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
        print(response)

        score = best_subspan_em(response, jsonObj["answers"][i])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": jsonObj["question"][i], "response": response, "gold_answer": jsonObj["answers"][i], "Score": score})
        print("Correct progress", correct_num)

    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/shuffle/sum_{reencode_num}/NQ_ckpt{ckpt}_at{pos}_{accuracy}_{time_str}.jsonl"
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
