import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import pandas as pd    
import json
import datetime
from rouge_score import rouge_scorer
from datasets import load_dataset

global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")

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

def inference(input_ids, past_key_values, model_name="meta-llama/Llama-2-7b-chat-hf", max_length=2000, num_return_sequences=1):

    tokenizer = global_tokenizer
    model = global_model
    
    model.eval()

    max_length = input_ids.size(1) + 400

    with torch.no_grad():

        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
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

def calculate_rouge_l_score(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    rouge_l_score = scores['rougeL'].fmeasure
    return rouge_l_score

def main():

    dataset = load_dataset("MemGPT/MSC-Self-Instruct")

    # print(reorganize_dialog(dataset["train"]["previous_dialogs"][0]))

    template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant."
    # template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n"
    # template = "[INST] <<SYS>>\nYou're an assistant who answer the question with the knowledge provided in the prompt\n<</SYS>>\n\n"

    total_num = len(dataset["train"]["previous_dialogs"])
    print(total_num)
    res_list = []
    score_list = []

    for i in range(total_num):

        print("id:", str(i))
        memory_list = reorganize_dialog(dataset["train"]["previous_dialogs"][i])

        for j in range(len(memory_list)):
            memory_list[j] = str(memory_list[j])[1:-1]
        # print(memory_list[0])
        start_token = "<s>"
        end_token = "[/INST]"
        memory_list.insert(0, template)
        memory_list.insert(0, start_token)
        # memory_list.append(end_token)

        new_prompt = "Answer the User's question based on above conversations. 'User': '" + dataset["train"]["self_instruct"][i]["B"] + "'[/INST]"

        # print(new_prompt)

        seq = ""

        for st in memory_list:
            seq = seq + st

        seq = seq + new_prompt

        input_ids = global_tokenizer(seq, return_tensors="pt").input_ids
        input_ids = input_ids.to(global_model.device)

        generated_seq = inference(input_ids, None)
        response = generated_seq[0].split('[/INST]')[1]

        gold_answer = dataset["train"]["self_instruct"][i]["A"]
        score = calculate_rouge_l_score(response, gold_answer)

        print('score:', str(score))
        score_list.append(score)
        res_list.append({"score": str(score),"question": dataset["train"]["self_instruct"][i]["B"], "response": response, "gold_answer": gold_answer})
        

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    final_score = sum(score_list) / len(score_list)

    file_name = f"result/dialog/dialog_baseline_{final_score}_{time_str}.json"

    with open(file_name, 'w') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
