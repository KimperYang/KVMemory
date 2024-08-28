# import pandas as pd    
# jsonObj = pd.read_json(path_or_buf='nq-open-10_total_documents_gold_at_0.jsonl', lines=True)

# memory_list = []

# print(jsonObj["question"][0])
# for i in range(0,10):
#     memory_list.append(jsonObj["ctxs"][0][i]["text"])

# print(len(jsonObj))


########################
# from datasets import load_dataset

# dataset = load_dataset("openwebtext")

# # Check the structure of the dataset
# print(dataset)

# # Access a specific split (e.g., 'train') and view the first few samples
# print(dataset['train'][28]['text'])

###########################
# import pandas as pd    
# jsonObj = pd.read_json(path_or_buf='/home/jingbo/KVMemory/data/ifeval/input_data.jsonl', lines=True)

# memory_list = []

# print(jsonObj["prompt"][0])

# memory_list = jsonObj["prompt"][3].split(". ")

# start_token = "<s>"
# end_token = "[/INST]"
# # memory_list.insert(0, template)
# memory_list.insert(0, start_token)

# print(memory_list)

################
from datasets import load_dataset

dataset = load_dataset("MemGPT/MSC-Self-Instruct")

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
                "User": person_a_text,
                "Assistant": person_b_text
            })
    
    return organized_dialog

print(len(dataset["train"]["previous_dialogs"]))