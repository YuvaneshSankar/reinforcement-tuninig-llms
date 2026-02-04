# import torch
# print( torch.cuda.is_available() )

from datasets import load_dataset
from unsloth import FastLanguageModel

dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Format the query using a ternary operator
def format_query(example):
    # 1. Always start with the instruction
    query = f"### Instruction:\n{example['instruction']}\n\n"

    # 2. Only add the input section if it's not empty
    if example.get("input") and example["input"].strip():
        query += f"### Input:\n{example['input']}\n\n"

    # 3. Always end with the response header
    query += "### Response:"

    return {"query": query}

# Apply the formatting
dataset = dataset.map(format_query, batched=False)
# # Let's see the results
# for i in range(15):
#     print(f"--- Sample {i} ---")
#     print(dataset[i]["query"])
#     print("-" * 20)





# # Tokenize
# def tokenize_function(examples):
#     tokenized = tokenizer(
#         examples["query"],
#         truncation=True,
#         max_length=2048,
#         padding="max_length", # PPO often benefits from consistent shapes, but False is okay if collator handles it
#         return_tensors="pt"
#     )
#     return tokenized

# dataset = dataset.map(tokenize_function, batched=True)




for i in range(15):
    print(f"--- Sample {i} ---")
    print(dataset[i]["query"])
    print("-" * 20)