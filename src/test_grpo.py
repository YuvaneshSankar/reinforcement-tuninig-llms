from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification


model = AutoModelForCausalLM.from_pretrained("./models/sft/checkpoint-6000")



reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name).to("cuda")

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



trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(use_vllm=True,vllm_mode="colocate"),
    reward_funcs=reward_model,
    train_dataset=dataset,
)

trainer.train()