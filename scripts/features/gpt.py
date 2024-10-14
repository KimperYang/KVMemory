import json
import pandas as pd    
import datetime
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
# def completion_with_backoff_mcopenai_chatgpt(**kwargs):
#     while True:
#         try:
#             client = AzureOpenAI(
#                 api_version="2024-02-15-preview",
#                 azure_endpoint="https://chatgpt-0125.openai.azure.com/",
#                 api_key="dca4cf09329941098c51a8ca09c036ef",
#             )
#             result = client.chat.completions.create(
#                 model="chatgpt_0125",
#                 **kwargs,
#             )
#             break
#         except Exception as e:
#             reason = e.body['code']
#             if reason == 'content_filter':
#                 return None
#             time.sleep(3)
#     return result


file_path = "/home/jingbo/KVMemory/result/dialog/dialog_unfinetuned_noanswerdisabled_0.5231973231017353_20241009-230330.jsonl"
jsonObj = pd.read_json(path_or_buf=file_path, lines=True)

total = len(jsonObj)
correct = 0
wrong = 0
judge = []
for i in range(total):
    print(i)
    q = jsonObj["question"][i]
    gold_a = jsonObj["gold_answer"][i]
    generate_a = jsonObj["response"][i]

    messages=[
        {"role": "system", "content": "You're an helpful assistant."},
        {"role": "user", "content": f"Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG. The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example: Question: Do you remember what I got the last time I went to Hawaii? Gold answer: A shell necklace The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. For example, the following answers would be considered CORRECT: Generated answer (CORRECT): Oh yeah, that was so fun! I got so much stuff there, including that shell necklace. Generated answer (CORRECT): I got a ton of stuff... that surfboard, the mug, the necklace, those coasters too.. Generated answer (CORRECT): That cute necklace The following answers would be considered WRONG: Generated answer (WRONG): Oh yeah, that was so fun! I got so much stuff there, including that mug. Generated answer (WRONG): I got a ton of stuff... that surfboard, the mug, those coasters too.. Generated answer (WRONG): I’m sorry, I don’t remember what you’re talking about. Now it’s time for the real question: Question: {q} Gold answer: {gold_a} Generated answer: {generate_a} First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script."}
    ]

    response = completion_with_backoff_mcopenai(messages = messages, temperature = 0, max_tokens=50).choices[0].message.content
    if "CORRECT" in response:
        correct+=1
    if "WRONG" in response:
        wrong+=1
    
    judge.append(response)

resdict = {}
resdict["correct"] = correct
resdict["wrong"] = wrong
resdict["total"] = total
resdict['judge'] = judge
print("Correct Num: ",correct)
print("Wrong Num: ",wrong)
print("Total Num: ",total)

current_time = datetime.datetime.now()
time_str = current_time.strftime("%Y%m%d-%H%M%S")

file_name = f"result/dialog/LLMJudge/MSCJudge_unfinetuned_noanswerdisabled_{time_str}.json"

with open(file_name, 'w', encoding='utf-8') as file:
    json.dump(resdict, file, ensure_ascii=False, indent=4)

print(f"Dumped at {file_name}")