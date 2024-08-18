import pandas as pd    
jsonObj = pd.read_json(path_or_buf='nq-open-10_total_documents_gold_at_0.jsonl', lines=True)

for i in range(0,10):
    print(jsonObj["ctxs"][0][i]["text"])