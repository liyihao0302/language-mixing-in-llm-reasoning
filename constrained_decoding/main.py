from utils import generate_constrained_response, check_answer, enforce_switch_on_text_tokens, draw_token_with_stats

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

import pandas as pd
custom_cache_dir = "/workspaces/002/data/huggingface_cache/"
dataset_path = "/workspaces/002/data/datasets/multilingual/translated_math500.csv"
# Load tokenizer and model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#model_name = "Qwen/QwQ-32B-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#print("All special tokens:", tokenizer.special_tokens_map)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for lower memory use
    device_map="auto",          # Automatically selects the GPU
    trust_remote_code=True,
    cache_dir=custom_cache_dir)


df_dataset = pd.read_csv(dataset_path)
prompt = df_dataset.iloc[19]["chin_query"]
answer = df_dataset.iloc[19]["chin_answer"]
#prompt="计算123*456。"
#answer = "56088"

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


# Set model to evaluation mode
model.eval()
inputs = tokenizer(text, return_tensors="pt").to(model.device)
generated_text, token_count, token_ids, mask = generate_constrained_response(model, inputs, tokenizer, max_token = 4096, mode='no_EN', location=None, do_sample=False, temperature=1.0, top_p=1.0)

print("🧠 Generated Response:")
print("-" * 40)
print(generated_text)
print("-" * 40)
print(f"📏 Token Count: {token_count}")
correct = check_answer(generated_text, answer)
print(f"✅ Is the answer correct? {'Yes ✅' if correct else 'No ❌'}")
truncate_length = 513
mask = mask[:truncate_length]
corrects, token_counts = enforce_switch_on_text_tokens(mask, model, inputs, tokenizer, answer, max_token = 4096, mode='switch_to_EN', location=None, do_sample=False, temperature=1.0, top_p=1.0)
print("Corrects: ", corrects)
print("Token counts: ", token_counts)

tokens = [tokenizer.decode(token_id) for token_id in token_ids[0]][:truncate_length]
#print("Generated text: ", generated_text[:5])

draw_token_with_stats(corrects.astype(np.float32), tokens, mask, x_start=0.05, y_start=0.9, fontsize=20, max_width=0.95, ax=None, line_spacing=0.1, stats_cmap = 'RdYlGn', label='Accuracy')
draw_token_with_stats(token_counts, tokens, mask, x_start=0.05, y_start=0.9, fontsize=20, max_width=0.95, ax=None, line_spacing=0.1, stats_cmap = 'Reds', label='Token Count')
