import pandas as pd    
jsonObj = pd.read_json(path_or_buf='nq-open-10_total_documents_gold_at_0.jsonl', lines=True)

memory_list = []

print(jsonObj["question"][0])
for i in range(0,10):
    memory_list.append(jsonObj["ctxs"][0][i]["text"])

print(len(jsonObj))