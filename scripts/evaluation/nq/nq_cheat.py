import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import pandas as pd    
import json
import datetime
import string
from typing import List
from peft import PeftModel, PeftConfig
import regex

jsonObj = pd.read_json(path_or_buf='data/raw/nq/nq-open-10_0.jsonl', lines=True)
global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")

peft_config_path = "/mnt/data/jingbo/kv_dump_combine"  # Path to the directory where LoRA weights are stored
lora_config = PeftConfig.from_pretrained(peft_config_path)

global_model = PeftModel.from_pretrained(base_model, peft_config_path)
#!/usr/bin/env python3


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

def generate_kv_with_id(input_ids, p_id):

    with torch.no_grad():
        out = global_model(input_ids, position_ids = p_id)
        past_key_values = out.past_key_values

    return past_key_values

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

    return concatenated_past_key_values

def inference(input_ids, past_key_values, model_name="meta-llama/Llama-2-7b-chat-hf", max_length=2000):

    tokenizer = global_tokenizer
    model = global_model
    
    model.eval()

    max_length = input_ids.size(1) + 200

    with torch.no_grad():

        outputs = model.generate(
            input_ids=input_ids,
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

def main():
    global_model.to('cuda')
    # template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n"

    template = "<s> [INST] Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n\n"

    total_num = len(jsonObj)
    correct_num = 0
    res_list = []

    for i in range(total_num):

        print("Processing sample:", str(i))
        memory_list = [template]

        
        for j in range(0,10):
            title = jsonObj["ctxs"][i][j]["title"]
            text = jsonObj["ctxs"][i][j]["text"]
            memory_list.append(f"Document [{j+1}](Title: {title}) {text}"+"\n")

        new_prompt = "\n\nQuestion: " + jsonObj["question"][i] + "\nAnswer:[/INST]"

        kv_list = []
        id_list = []
        idx = 0

        for st in memory_list:
            # print(st)
            id = global_tokenizer(st, return_tensors="pt", add_special_tokens=False).input_ids
            position_id = torch.arange(idx, idx + id.size(1)).unsqueeze(0)
            # print(position_id[0])
            # print(id.size(1))
            kv = generate_kv_with_id(id.to(global_model.device), position_id.to(global_model.device))
            id_list.append(id)
            kv_list.append(kv)

            idx = idx + id.size(1)

        appended_kv = append_kv(kv_list)

        prompt_id = global_tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False).input_ids
        id_list.append(prompt_id)

        concat_id = torch.cat(id_list, dim=1).to(global_model.device)
        generated_seq = inference(concat_id, appended_kv)
        response = generated_seq[0].split('[/INST]')[1]
        print(response)

        score = best_subspan_em(response, jsonObj["answers"][i])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": jsonObj["question"][i], "response": response, "gold_answer": jsonObj["answers"][i], "Score": score})
        print("Correct progress", correct_num)
        
    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/nq/nq_combinecheat_at0_{accuracy}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
