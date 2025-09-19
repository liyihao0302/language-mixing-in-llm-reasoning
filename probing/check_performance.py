import pandas as pd
import json
from utils.check_answer import evaluate_answer, extract_after_boxed
dataset_name = 'math500'
unconstrained_file = '/workspaces/002/data/cot_outputs/'+dataset_name+'_qwq32b_preview_responses_max4096_unconstrain.jsonl'
constrained_probe_file = '/workspaces/002/data/cot_outputs/'+dataset_name+'_decoding_w_probe_outputs.json'

dataset_file = '/workspaces/002/data/datasets/multilingual/translated_'+dataset_name+'.csv'
df = pd.read_csv(dataset_file)
unconstrained_data = []
with open(unconstrained_file, 'r') as f:
    for line in f:
        unconstrained_data.append(json.loads(line))

# Read the constrained file (json format)
with open(constrained_probe_file, 'r') as f:
    constrained_data = json.load(f)
uncon_df = pd.DataFrame(unconstrained_data)

con_df = pd.DataFrame(constrained_data)

sample_idx = con_df['sample_idx'].values

# 1. Select the matched rows
selected_con_df = con_df
selected_uncon_df = uncon_df.iloc[sample_idx]
selected_df = df.iloc[sample_idx]
probe_effective = 0
# 2. Loop through the matched rows
for i in range(len(selected_con_df)):
    
    #import pdb; pdb.set_trace()
    # 3. Extract Chinese responses
    
    chin_response_con = selected_con_df.iloc[i].get('generated_text')
    chin_response_uncon = selected_uncon_df.iloc[i].get('chin_response')
    answer = selected_df.iloc[i].get('answer')
    #import pdb; pdb.set_trace()
    # 4. Evaluate
    is_correct_con = evaluate_answer(answer, extract_after_boxed(chin_response_con))
    is_correct_uncon = evaluate_answer(answer, extract_after_boxed(chin_response_uncon))
    #import pdb; pdb.set_trace()

    if selected_con_df.iloc[i].get('probe_log') != []:
        print(f"Sample {i}: gt_answer: {answer}, sample_idx: {selected_con_df.iloc[i].get('sample_idx')}")
        print(f"  Probe log: {selected_con_df.iloc[i].get('probe_log')}")
        print(f"  Constrained correct? {is_correct_con}, answer: {extract_after_boxed(chin_response_con)}")
        print(f"  Unconstrained correct? {is_correct_uncon}, answer: {extract_after_boxed(chin_response_uncon)}")
        
        probe_effective += 1

print(f"Probe effective: {probe_effective}/{len(selected_con_df)}")