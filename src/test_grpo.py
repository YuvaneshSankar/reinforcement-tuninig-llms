from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

# Load base model and adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/tinyllama-bnb-4bit",
    device_map="auto",
    trust_remote_code=True
)

# Load your PEFT adapter checkpoint
model = PeftModel.from_pretrained(
    base_model,
    "Yuvanesh123/grpo_test_checkpoint-6000"
)

# Reward model
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_name,
    device_map="auto"
)

# Dataset formatting
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_query(example):
    query = f"### Instruction:\n{example['instruction']}\n\n"

    if example.get("input") and example["input"].strip():
        query += f"### Input:\n{example['input']}\n\n"

    query += "### Response:"

    return {"prompt": query}

dataset = dataset.map(format_query, batched=False)

# GRPO Training
trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        num_generations=32,
        num_iterations=2,
        generation_batch_size=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        num_train_epochs=2,
        max_steps=1000,
        learning_rate=2e-5,
        max_completion_length=1024,
    ),
    reward_funcs=reward_model,
    train_dataset=dataset,
)

trainer.train()

import matplotlib.pyplot as plt
import pandas as pd

history = pd.DataFrame(trainer.state.log_history)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

if 'loss' in history.columns:
    axes[0].plot(history['step'], history['loss'], label='Train Loss', color='blue')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Steps')

if 'reward' in history.columns:
    axes[1].plot(history['step'], history['reward'], label='Reward', color='green')
    axes[1].set_title('Mean Reward')
    axes[1].set_xlabel('Steps')

if 'kl' in history.columns:
    axes[2].plot(history['step'], history['kl'], label='KL Div', color='red')
    axes[2].set_title('KL Divergence')
    axes[2].set_xlabel('Steps')

plt.tight_layout()
plt.savefig('training_metrics.png')
print("Graphs saved to training_metrics.png")
