import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import pandas as pd    
import json
import datetime
from rouge_score import rouge_scorer
from datasets import load_dataset
from openai import AzureOpenAI

def completion_with_backoff_mcopenai(**kwargs):
    client = AzureOpenAI(
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        api_version="2023-12-01-preview",
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint="https://0212openai.openai.azure.com/",
        api_key="352d7f1511084d6d8a37f7214c5eb528",
    )
    result = client.chat.completions.create(
        model="gpt4-azure-0212",
        **kwargs,
    )
    return result

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
        memory_list = []
        print("id:", str(i))
        # memory_list.append(template)

        for j in range(len(dataset["train"]["summary_speaker_1"][i])):
            # print(j)
            memory = reorganize_summary(dataset["train"]["summary_speaker_1"][i][j], dataset["train"]["summary_speaker_2"][i][j])
            memory_list.append(memory)

        memory = " ".join(memory_list)
        template = f"Your task is to answer a question from the user about your prior conversations. The following is a summary of all your prior conversations: {memory} Answer from the perspective of the persona provided (do not say that you are an AI assistant)."
        # print(memory_list)
        # print(new_prompt)

        seq = dataset["train"]["self_instruct"][i]["B"]
        # print(seq)
        messages=[
            {"role": "system", "content": template},
            {"role": "user", "content": seq}
        ]
        response = completion_with_backoff_mcopenai(messages = messages, temperature = 0, max_tokens=100).choices[0].message.content

        gold_answer = dataset["train"]["self_instruct"][i]["A"]
        score = calculate_rouge_l_score(response, gold_answer)

        print('answer', response)
        print('score:', str(score))
        score_list.append(score)
        res_list.append({"score": str(score),"question": dataset["train"]["self_instruct"][i]["B"], "response": response, "gold_answer": gold_answer})
        

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    final_score = sum(score_list) / len(score_list)

    file_name = f"result/dialog/dialog_gpt_longer_{final_score}_{time_str}.json"

    with open(file_name, 'w') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
