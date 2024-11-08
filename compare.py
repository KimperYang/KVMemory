import json
import pandas as pd    
import datetime

path1 = 'result/nq/nq_llama3.21b_original_0_0.544_20241017-013003.jsonl'
path2 = 'result/11-3/nq/nq_llama3.2_1B_upper_warmup0.1_decaycosine_0steps_at9_0.458_20241106-194046.jsonl'

jsonObj1 = pd.read_json(path_or_buf=path1, lines=True)
jsonObj2 = pd.read_json(path_or_buf=path2, lines=True)

for i in range(500):
    if(jsonObj1["response"][i] != jsonObj2["response"][i]):
        print(i)
        print(jsonObj1["response"][i])
        print(jsonObj2["response"][i])
        break