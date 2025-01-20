import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import pandas as pd    
import json
import datetime
import string
from typing import List
from datasets import load_from_disk
from src.data.compress import insert_mem_tokens, get_position_id, construct_compress_attention_matrix
import regex
import transformers.models.llama.modeling_llama
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

global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")

global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)

# vocab_size = len(global_tokenizer)
# base_model.resize_token_embeddings(vocab_size)

# peft_config_path = "/mnt/data/jingbo/kv_dump_combine_mix5_5000steps_5e-6_full/checkpoint-5000"  # Path to the directory where LoRA weights are stored

# global_model = PeftModel.from_pretrained(base_model, peft_config_path)

def filter_id(input_ids, intervals_to_remove):  

    T = input_ids.shape[1] 
    mask = torch.ones(T, dtype=bool)

    for (start, end) in intervals_to_remove:
        mask[start:end] = False  # set these indices to False

    return input_ids[:, mask]

def filter_kv(past_key_values, intervals_to_remove):
    num_layers = len(past_key_values)
    filtered_past_key_values = ()    

    T = past_key_values[0][0].shape[2] 
    mask = torch.ones(T, dtype=bool)

    for (start, end) in intervals_to_remove:
        mask[start:end] = False  # set these indices to False

    for layer_id in range(num_layers):
        tem_key = past_key_values[layer_id][0]
        tem_value = past_key_values[layer_id][0]

        filtered_key = tem_key[:, :, mask, :]
        filtered_value = tem_value[:, :, mask, :]

        filtered_past_key_values += ((filtered_key, filtered_value),)

    return filtered_past_key_values

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

    mem_start = 128254
    mem_end = 128255
    compress_tokens = list(range(128011, 128031))

    global_model.to('cuda')
    # template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n"

    total_loss = 0

    dataset = load_from_disk("dataset_cache/processed/block_qa/qa_mem")['test']
    total_num = 1

    for i in range(total_num):

        template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        sys_id = global_tokenizer(template, add_special_tokens=False).input_ids
        print(len(sys_id))
        print("Processing sample:", str(i))
        example = dataset[i]

        raw_input_ids = sys_id
        position = len(sys_id)
        biased_ranges = []

        for j in range(0,10):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = global_tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
            raw_input_ids += tem_id
            biased_ranges.append([position, position + len(tem_id)])

            position += len(tem_id)

        # new_ids, new_ranges = insert_mem_tokens(
        #     raw_input_ids, biased_ranges, compress_tokens, mem_start, mem_end
        # )

        # position_ids = get_position_id(new_ids, new_ranges)

        # attention_matrix = construct_compress_attention_matrix(len(new_ids), new_ranges, len(new_ids), global_model.device).unsqueeze(0).unsqueeze(0)       

        # new_ids = torch.tensor([new_ids], device = global_model.device)
        # position_ids = torch.tensor([position_ids], device = global_model.device)

        # with torch.no_grad():
        #     outputs = global_model(input_ids = new_ids, attention_mask = attention_matrix, position_ids = position_ids)
        #     past_key_values = outputs.past_key_values

        # filtered_kv = filter_kv(past_key_values, new_ranges)
        # filtered_id = filter_id(new_ids, new_ranges)

        # new_prompt = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        # prompt_id = global_tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(global_model.device)

        # generate_id = torch.cat([filtered_id, prompt_id], dim = 1)

        # with torch.no_grad():

        #     outputs = global_model.generate(
        #         input_ids=generate_id,
        #         max_new_tokens=200,
        #         do_sample=False,
        #         temperature=None,
        #         top_p=1.0,
        #         past_key_values=filtered_kv,
        #         use_cache=True
        #     )
        # # print(outputs)
        # generated_seq = global_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # response = generated_seq[0].split('assistant\n\n')[-1]
        # print(response)

# Cache and calculate loss
        new_ids, new_ranges = insert_mem_tokens(
            raw_input_ids, biased_ranges, compress_tokens, mem_start, mem_end
        )
        # print(new_ranges)
        position_ids = get_position_id(new_ids, new_ranges)

        attention_matrix = construct_compress_attention_matrix(len(new_ids), new_ranges, len(new_ids), global_model.device).unsqueeze(0).unsqueeze(0)       

        new_ids = torch.tensor([new_ids], device = global_model.device)
        position_ids = torch.tensor([position_ids], device = global_model.device)

        # print(new_ids)
        print(attention_matrix[0][0][132])
        with torch.no_grad():
            outputs = global_model(input_ids = new_ids, attention_mask = attention_matrix, position_ids = position_ids)
            past_key_values = outputs.past_key_values

        filtered_kv = filter_kv(past_key_values, new_ranges)
        filtered_id = filter_id(new_ids, new_ranges)
        # filtered_pos = filter_id(position_ids, new_ranges)
        # print(filtered_pos)
        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = global_tokenizer(user, add_special_tokens=False).input_ids

        ans_id = global_tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids

        input_ids = user_id + ans_id
        labels = [-100] * len(user_id) + ans_id

        with torch.no_grad():

            outputs2 = global_model(
                input_ids=torch.tensor([input_ids], device = global_model.device),
                labels=torch.tensor([labels], device = global_model.device),
                past_key_values=filtered_kv,
                use_cache=True
            )
            
            total_loss += outputs2.loss.item()

        print(f"Avg loss {total_loss / (i+1)}")


# # Calculate loss directly

#         new_ids, new_ranges = insert_mem_tokens(
#             raw_input_ids, biased_ranges, compress_tokens, mem_start, mem_end
#         )

#         user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
#         user_id = global_tokenizer(user, add_special_tokens=False).input_ids
#         new_ids += user_id

#         ans_id = global_tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False).input_ids
#         new_ids += ans_id

#         position_ids = get_position_id(new_ids, new_ranges)

#         attention_matrix = construct_compress_attention_matrix(len(new_ids), new_ranges, len(new_ids), global_model.device).unsqueeze(0).unsqueeze(0)       

#         labels = [-100] * (len(new_ids) - len(ans_id)) + ans_id
#         labels = torch.tensor([labels], device = global_model.device)

#         print(labels)

#         new_ids = torch.tensor([new_ids], device = global_model.device)
#         position_ids = torch.tensor([position_ids], device = global_model.device)

#         with torch.no_grad():
#             outputs = global_model(input_ids = new_ids, attention_mask = attention_matrix, position_ids = position_ids, labels=labels)
#             total_loss += outputs.loss.item()

#         print(f"Avg loss {total_loss / (i+1)}")
    
    print(f"Final avg loss {total_loss / total_num}")

if __name__ == "__main__":
    main()
