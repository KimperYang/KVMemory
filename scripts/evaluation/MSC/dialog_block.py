import argparse
import datetime
import json

import pandas as pd
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.attention import construct_biased_attention_matrix

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--run', type=str, required=True, help='Path under training_res')
parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')

args = parser.parse_args()

run_name = args.run
ckpt = args.ckpt

if "meta" not in run_name:
    global_tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")

    global_model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)

else:
    global_tokenizer = AutoTokenizer.from_pretrained(f"{run_name}")

    global_model = AutoModelForCausalLM.from_pretrained(f"{run_name}", torch_dtype=torch.bfloat16)

# vocab_size = len(global_tokenizer)
# base_model.resize_token_embeddings(vocab_size)

# peft_config_path = "/mnt/data/jingbo/kv_dump_combine_mix5/checkpoint-5000"  # Path to the directory where LoRA weights are stored
# lora_config = PeftConfig.from_pretrained(peft_config_path)

# global_model = PeftModel.from_pretrained(base_model, peft_config_path)

def generate_kv(prompt):

    tokenizer = global_tokenizer
    model = global_model
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # out = model(**inputs, use_cache=True)
    # print('device',model.device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # print(input_ids)
    input_ids = input_ids.to(model.device)
    out = model(input_ids)
    past_key_values = out.past_key_values

    #filter <s>
    filtered_past_key_values = ()

    for past_keys, past_values in past_key_values:

        filtered_keys = past_keys[:, :, 1:, :] 
        filtered_values = past_values[:, :, 1:, :] 
        filtered_past_key_values = filtered_past_key_values + ((filtered_keys, filtered_values),)

    input_ids = input_ids[:, 1:]

    # print(filtered_past_key_values[0][0].size())
    # print(filtered_past_key_values.get_seq_length())

    return input_ids, filtered_past_key_values

def append_kv(kv_list):
    if not kv_list:
        raise ValueError("kv_list is empty. It must contain at least one past_key_values list.")

    num_layers = len(kv_list[0])

    concatenated_past_key_values = ()

    for layer in range(num_layers):

        keys_list = [kv[layer][0] for kv in kv_list]
        values_list = [kv[layer][1] for kv in kv_list]

        concatenated_keys = torch.cat(keys_list, dim=2)
        concatenated_values = torch.cat(values_list, dim=2) 

        concatenated_past_key_values = concatenated_past_key_values + ((concatenated_keys, concatenated_values),)
    # keys, values = concatenated_past_key_values[0], concatenated_past_key_values[1]
    # torch.save(keys, "keys.pt")
    # torch.save(values, "values.pt")
    return concatenated_past_key_values

def inference(input_ids, past_key_values, model_name="meta-llama/Llama-3.2-1B-Instruct", max_length=2000):

    tokenizer = global_tokenizer
    model = global_model
    attention_msk = torch.tensor([[1]*(input_ids.size(1) + past_key_values[0][0].size(2))]).to(input_ids.device)
    model.eval()

    max_length = input_ids.size(1) + 100

    with torch.no_grad():

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_msk,
            max_length=max_length,
            do_sample=False,
            temperature=None,
            top_p=1.0,
            past_key_values=past_key_values,
            use_cache=True
        )
    # print(outputs)
    generated_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return generated_sequences

def count_tokens(input_text):

    tokenizer = global_tokenizer

    tokens = tokenizer.encode(input_text, add_special_tokens=True)

    num_tokens = len(tokens)
    for token_id in tokens:
        token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        print(f"Token ID: {token_id}, Token: '{token}'")
    print(f"Number of tokens including special tokens: {num_tokens}")

def reorganize_dialog(data):
    organized_dialog = []
    
    for entry in data:
        dialog = entry['dialog']
        
        # Assume alternating text between PersonA and PersonB
        for i in range(0, len(dialog) - 1, 2):
            person_a_text = dialog[i]['text']
            person_b_text = dialog[i + 1]['text']
            
            # Append each exchange as a dictionary entry
            organized_dialog.append({
                "Assistant": person_a_text,
                "User": person_b_text
            })
    
    return organized_dialog

def reorganize_summary(sum1, sum2):

    concatenated_asst = "You: " + " ".join(sum1)
    concatenated_user = "User: " + " ".join(sum2)

    return concatenated_asst + " " + concatenated_user

def calculate_rouge_l_score(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    rouge_l_score = scores['rougeL'].recall
    return rouge_l_score

def main():
    global_model.to('cuda')
    dataset = load_dataset("MemGPT/MSC-Self-Instruct")

    # print(reorganize_dialog(dataset["train"]["previous_dialogs"][0]))

    # template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n"
    # template = "[INST] <<SYS>>\nYou're an assistant who answer the question with the knowledge provided in the prompt\n<</SYS>>\n\n"

    data = dataset["train"]["summary_speaker_1"]
    total_num = len(data)
    print(total_num)
    res_list = []
    score_list = []

    for i in range(total_num):
        memory_list = ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYour task is to answer a question from the user about your prior conversations.<|eot_id|>"]
        print("id:", str(i))

        for j in range(len(dataset["train"]["summary_speaker_1"][i])):
            # print(j)
            memory = reorganize_summary(dataset["train"]["summary_speaker_1"][i][j], dataset["train"]["summary_speaker_2"][i][j])
            memory_list.append(memory)

        current_position = 0
        id_list = []
        biased_index = []

        for st in memory_list:
            tem_id = global_tokenizer(st, add_special_tokens=False, return_tensors="pt").input_ids

            id_list.append(tem_id)

            if "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" not in st:
                biased_index.append([current_position, current_position + tem_id.size(1)])

            current_position = current_position + tem_id.size(1)

        question = "<|start_header_id|>user<|end_header_id|>\n\n Answer from the perspective of the conversation summaries provided (do not say that you are an AI assistant)." + dataset["train"]["self_instruct"][i]["B"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        question_ids = global_tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids

        cache_id = torch.cat(id_list, dim=1)

        generate_id = torch.cat([cache_id, question_ids], dim=1).to(global_model.device)

        attention_matrix = construct_biased_attention_matrix(cache_id.size(1), biased_index, cache_id.size(1), global_model.device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            outputs = global_model(input_ids = cache_id.to(global_model.device), attention_mask = attention_matrix)
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

        generated_seq = global_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = generated_seq[0].split("assistant\n\n")[-1]

        gold_answer = dataset["train"]["self_instruct"][i]["A"]
        score = calculate_rouge_l_score(response, gold_answer)

        print('answer', response)
        print('score:', str(score))
        score_list.append(score)
        res_list.append({"score": str(score),"question": dataset["train"]["self_instruct"][i]["B"], "response": response, "gold_answer": gold_answer})

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    final_score = sum(score_list) / len(score_list)

    if "meta" not in run_name:
        file_name = f"result/new_data/block/MSC_ckpt{ckpt}_{final_score}_{time_str}.json"
    else:
        file_name = f"result/new_data/block/MSC_promptcache_ckpt{ckpt}_{final_score}_{time_str}.json"

    with open(file_name, 'w') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
