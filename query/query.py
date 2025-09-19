import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

OUTPUT_FILE = '/workspaces/002/data/cot_outputs/gaokao_cloze_deepseekr1_distill_qwen1.5b_responses_max2048.jsonl'
dataset_path = "/workspaces/002/data/datasets/multilingual/translated_gaokao_cloze.csv"
custom_cache_dir = "/workspaces/002/data/huggingface_cache/"
# Load tokenizer and model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for lower memory use
    #device_map="auto",          # Automatically selects the GPU
    trust_remote_code=True,
    cache_dir=custom_cache_dir).to('cuda:0')

def generate_response(question, max_tokens=2048):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            top_p = None,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
        )

    # Remove the input tokens to get only the generated part
    new_tokens = outputs[0][len(inputs.input_ids[0]):]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return decoded.strip()


# === Load Dataset ===
print("Loading dataset...")
df = pd.read_csv(dataset_path)
df.head(2)

# === Main Loop ===
print("Starting generation...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        chin_prompt = row["chin_query"]
        eng_prompt = row["query"]
        problem_id = idx  # or any unique ID you prefer

        try:
            chin_response = generate_response(chin_prompt)
        except Exception as e:
            chin_response = f"[ERROR] {e}"

        try:
            eng_response = generate_response(eng_prompt)
        except Exception as e:
            eng_response = f"[ERROR] {e}"

        record = {
            "problem_id": problem_id,
            "chin_prompt": chin_prompt,
            "chin_response": chin_response,
            "eng_prompt": eng_prompt,
            "eng_response": eng_response,
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


print("Done. Output saved to:", OUTPUT_FILE)

