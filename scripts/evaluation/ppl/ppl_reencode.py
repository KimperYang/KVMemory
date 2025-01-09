import argparse
import datetime
import json
import math
import random

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.attention import construct_biased_attention_matrix
from transformers.models.llama import modeling_llama
parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--run', type=str, required=True, help='Path under training_res')
parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')
parser.add_argument('--reencode', type=int, required=True, help='Reencode number')

args = parser.parse_args()

run_name = args.run
ckpt = args.ckpt
reencode_num = args.reencode

global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")

global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)

def main():
    global_model.to('cuda')
    # template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n"
    dataset = load_from_disk('dataset_cache/processed/fineweb/text_mem')['test']

    processed = 0
    ppl = 0
    max_len = 4096
    random.seed(42)

    for i in range(len(dataset)):

        # if processed > 1:
        #     break

        input_ids = global_tokenizer(dataset[i]['text'], add_special_tokens= False)['input_ids']

        if len(input_ids) < 1000:
            continue

        sys = "<|begin_of_text|>"
        sys_tokens = global_tokenizer(sys, add_special_tokens= False)['input_ids']
        sys_len = len(sys_tokens)

        user_tokens = []
        user_len = len(user_tokens)

        text = dataset[i]["text"]
        input_ids = global_tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:max_len - user_len - sys_len]
        mem_num = random.randint(5,40)

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - (2 + reencode_num) * mem_num]

        mem_len = len(input_ids) - 128

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]
        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        biased_index = []
        bias_position = sys_len

        for i in range(mem_num):
            tem_mem_id = [128256] + split_memory_ids[i] + [128257] + [128258] * reencode_num
            concat_ids += tem_mem_id

            biased_index.append([bias_position, bias_position + len(tem_mem_id) - reencode_num])
            bias_position = bias_position + len(tem_mem_id)

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + (2 + reencode_num) *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids
        attention_matrix = construct_biased_attention_matrix(len(concat_ids), biased_index, len(concat_ids), global_model.device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = global_model(input_ids = torch.tensor([concat_ids]).to(global_model.device), attention_mask = attention_matrix, labels = torch.tensor([labels]).to(global_model.device))
            loss = output.loss

        tem_ppl = math.exp(loss.item())

        ppl += tem_ppl
        processed += 1
        print("processed:", processed)

    avg_ppl = ppl / processed
    print(avg_ppl)

    # current_time = datetime.datetime.now()
    # time_str = current_time.strftime("%Y%m%d-%H%M%S")

    # file_name = f"result/new_data/bias/ppl_ckpt{ckpt}_{avg_ppl}_{time_str}.jsonl"

    # with open(file_name, 'w', encoding='utf-8') as f:
    #     for entry in res_list:
    #         json_line = json.dumps(entry)
    #         f.write(json_line + '\n')

    # print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
