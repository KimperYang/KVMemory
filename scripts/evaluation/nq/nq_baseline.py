import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import pandas as pd    
import json
import datetime

jsonObj = pd.read_json(path_or_buf='data/nq/nq-open-10_9.jsonl', lines=True)
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

    print(filtered_past_key_values[0][0].size())
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

def check_result(ans_list, response):
    for ans in ans_list:
        if ans in response: 
            print("Response: ", response, "\nTRUE")
            return True

    print("Response: ", response, "\nFALSE")
    return False

def main():
    # template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n"

    template = "[INST] <<SYS>>\nYou're an assistant who answer the question with the knowledge provided in the prompt\n<</SYS>>\n\n"

    total_num = len(jsonObj)
    correct_num = 0
    res_dict = {}

    for i in range(total_num):

        print("id:", str(i))
        memory_list = []

        for j in range(0,10):
            memory_list.append(jsonObj["ctxs"][i][j]["text"])

        start_token = "<s>"
        end_token = "[/INST]"
        memory_list.insert(0, template)
        memory_list.insert(0, start_token)
        # memory_list.append(end_token)

        new_prompt = "Question: " + jsonObj["question"][i] + "[/INST]"

        kv_list = []
        id_list = []
        seq = ""

        for st in memory_list:
            seq = seq + st + "\n"

        seq = seq + new_prompt

        input_ids = global_tokenizer(seq, return_tensors="pt").input_ids
        input_ids = input_ids.to(global_model.device)

        generated_seq = inference(input_ids, None)
        response = generated_seq[0].split('[/INST]')[1]
        res_dict["res_"+str(i)] = response
        print(response)

        if check_result(jsonObj["answers"][i], response):
            res_dict["score_"+str(i)] = "TRUE"
            correct_num = correct_num + 1
            print("TRUE")

        else:
            res_dict["score_"+str(i)] = "FALSE"
            print("FALSE")
        
        print("progress", correct_num)
        
    res_dict["Correct Number"] = str(correct_num)
    res_dict["Total Number"] =str(total_num)
    res_dict["Accuracy"] = str(correct_num / total_num)
    print(correct_num / total_num)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/baseline_9_{time_str}.json"

    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(res_dict, file, ensure_ascii=False, indent=4)

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
