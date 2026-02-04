from trl import PPOConfig, PPOTrainer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset




model = AutoModelForCausalLM.from_pretrained("unsloth/tinyllama-bnb-4bit")



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

ppo_config = PPOConfig(
    output_dir="./models/ppo_vllm",
    use_vllm=True,
    kl_coef=0.05,
    gamma=1,
    lam=0.95,
    cliprange_value=0.2,
    vf_coef=0.1,
    mini_batch_size=1, # Adjust based on your GPU VRAM (start small)
    gradient_accumulation_steps=1,
)

ppo_trainer = PPOTrainer(
    args=ppo_config,
    model=model,
    ref_model=None,     # Unsloth handles ref_model efficiently internally
    value_model=model,
    train_dataset=dataset,
    reward_model=reward_model,
)

ppo_trainer.train(resume_from_checkpoint="./models/sft/checkpoint-6000")