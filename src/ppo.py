import unsloth
from unsloth import FastLanguageModel
# Patching must happen before other imports for some versions, though Unsloth handles this well now.
from trl import PPOTrainer, PPOConfig,AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
import torch

# 1. Setup Reward Model
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name)

def ppo_training():
    # 2. Load Model + SFT Adapters
    # Point 'model_name' to your SFT checkpoint folder.
    # Unsloth loads the base model & attaches the adapter automatically.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./models/sft/checkpoint-6000",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # 3. PPO Specific Setup
    # Generation (Rollout) requires LEFT padding. SFT uses right.
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Enable gradients for training
    FastLanguageModel.for_training(model)
    # 4. Prepare Dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Format the query
    dataset = dataset.map(lambda example: {
        "query": f"### Instruction {example['instruction']} \n### Input: {example['input']} \n### Response:"
    }, batched=False)

    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["query"],
            truncation=True,
            max_length=2048,
            padding="max_length", # PPO often benefits from consistent shapes, but False is okay if collator handles it
            return_tensors="pt"
        )
        return tokenized

    dataset = dataset.map(tokenize_function, batched=True)

    # Remove raw text columns to avoid collation errors
    dataset = dataset.remove_columns(['instruction', 'input', 'output', 'text', 'query'])
    dataset.set_format('torch')

    # 5. PPO Config
    ppo_config = PPOConfig(
        kl_coef=0.05,
        gamma=1,
        lam=0.95,
        cliprange_value=0.2,
        vf_coef=0.1,
        mini_batch_size=1, # Adjust based on your GPU VRAM (start small)
        gradient_accumulation_steps=1,
    )

    # 6. Initialize Trainer
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=None,     # Unsloth handles ref_model efficiently internally
        processing_class=tokenizer,
        value_model=None,
        train_dataset=dataset,
        reward_model=reward_model,
    )

    # 7. Start Training
    ppo_trainer.train()

if __name__ == "__main__":
    ppo_training()