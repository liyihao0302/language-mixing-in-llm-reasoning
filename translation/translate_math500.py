import re
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd

from openai import OpenAI
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

client = OpenAI(
    api_key="sk-RNTGuhA3o49NFaUNVStYFLqQhUASCWGQmfBqnxernKCZL1R8",
    base_url="https://api.456478.xyz/v1"
)


def translate_text(text):
    text = str(text)
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"You are a professional English-Chinese translater. You're asked to translate the texts of math problems from English to Chinese closely following the original text format, only changing the English characters and keep the mathematical representation as close as possible. Avoid inserting any additional spaces and line breaks. Do not try to solve the problem. Only do the translation. Translate as closely as possible."},
            {"role":"user","content":f"Please translate the following to Chinese: {text}"}
        ],
        temperature=0
    )
    return res.choices[0].message.content.strip()

rows = []
for row in tqdm(ds, total=len(ds)):
    subject, level, id, orig_q, orig_a = row['subject'], row['level'], row["unique_id"], row["problem"], row["answer"]
    q = translate_text(orig_q)
    a = translate_text(orig_a) if re.search(r"[A-Za-z]", str(orig_a)) else orig_a
    rows.append({"chin_query": q, "chin_answer": a, "query": orig_q, "answer": orig_a, "subject":subject, "level":level, "problem_id": id})

translated_ds_cloze = pd.DataFrame(rows)
translated_ds_cloze.to_csv("translated_math500.csv", index=False)
