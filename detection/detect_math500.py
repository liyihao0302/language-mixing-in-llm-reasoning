import json
from transformers import AutoTokenizer
import pandas as pd
from utils import vis_question_answer, detect_code_switching
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

custom_cache_dir = "/workspaces/002/data/huggingface_cache/"#huggingface cache dir
dataset_path = "/workspaces/002/data/datasets/multilingual/translated_math500.csv"
cot_output_path = "/workspaces/002/data/cot_outputs/math500_qwq32b_preview_responses_max4096_unconstrain.jsonl"
switch_detection_path = "math500_qwq32b_preview_code_switching.csv"


model_name = "Qwen/QwQ-32B-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=custom_cache_dir)


with open(cot_output_path, "r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]
# dict_keys(['problem_id', 'subject', 'level', 'chin_prompt', 'chin_response', 'eng_prompt', 'eng_response'])


df_dataset = pd.read_csv(dataset_path)
#Index(['chin_query', 'chin_answer', 'query', 'answer', 'subject', 'level','problem_id'],
answers = df_dataset["answer"].tolist()

target_idx = 303
code_switching_list_ch = []
code_switching_list_en = []
truncated_ch = []
truncated_en = []

# save into a csv file with 'problem_id', 'switch_count', 'token_length','subject', 'level', 'switch_snippets'
with open(switch_detection_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['problem_id', 'switch_count_ch', 'switch_snippets_ch', 'switch_count_en', 'switch_snippets_en'])
    writer.writeheader()


    for idx, (result, answer) in tqdm(enumerate(zip(results, answers))):
        
        ch_output = result["chin_response"]
        en_output = result["eng_response"]
        problem_id = result["problem_id"]
        ch_token_count = result["chin_token_count"]
        en_token_count = result["eng_token_count"]
        #subject = result["subject"]
        #level = result["level"]
        #answer = answer

        # Visualize the question, output, and answer
        #if idx == target_idx:
        #    vis_question_answer(question, output, answer)
        

        if ch_token_count == 4096:
            truncated_ch.append(idx)
        if en_token_count == 4096:
            truncated_en.append(idx)
        ch_count, ch_snippets = detect_code_switching(ch_output)
        if ch_count>=2:
            code_switching_list_ch.append(idx)
        #token_length = len(tokenizer.tokenize(output))

        ch_snippet_str = ""
        for snip in ch_snippets:
            
            # snippets = [{"position": ..., "snippet": "..."}, ...]
            ch_snippet_str = '|'.join(f"{i+1}. {snip}\n" for i, snip in enumerate(ch_snippets))

        en_count, en_snippets = detect_code_switching(en_output)
        if en_count>0:
            code_switching_list_en.append(idx)
        en_snippet_str = ""

        snippet_str = ""
        for snip in en_snippets:
            # snippets = [{"position": ..., "snippet": "..."}, ...]
            en_snippet_str = '|'.join(f"{i+1}. {snip}\n" for i, snip in enumerate(en_snippets))
        # Write one row per problem
        writer.writerow({
            'problem_id': idx,
            'switch_count_ch': ch_count,
            'switch_snippets_ch': ch_snippet_str,
            'switch_count_en': en_count,
            'switch_snippets_en': en_snippet_str,
            #'subject': subject,
            #'level': level,
        })
        
print('Chinese code switching problems: ', code_switching_list_ch)
print('English code switching problems: ', code_switching_list_en)
print('Truncated Chinese problems: ', truncated_ch)
print('Truncated English problems: ', truncated_en)
print(len(code_switching_list_ch))
