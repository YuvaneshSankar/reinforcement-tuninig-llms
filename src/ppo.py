import torch
from trl import PPOTrainer , PPOConfig
from unsloth import FastLanguageModel
from peft import PeftModel
from rm import get_reward_Score


def load_model_and_tokenizer(max_seq_length ,model_name):
  base_model,tokenizer=FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
  )

  return base_model,tokenizer


def load_adapter(base_model, adapter_dir):
  model=PeftModel.from_pretrained(base_model, adapter_dir)
  return model


def ppo_training(datasets):

  base_model,tokenizer=load_model_and_tokenizer(2048,"unsloth/tinyllama-bnb-4bit")
  model=load_adapter(base_model,"./models/sft/checkpoint-6000")
  ref_model=model.clone()

  ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    init_kl_coef=0.2,
    cliprange=0.2
  )


  ppo_trainer = PPOTrainer(
      config=ppo_config,
      model=model,
      ref_model=ref_model,
      tokenizer=tokenizer
  )

  for prompt in datasets:
    responses=ppo_trainer.generate(prompt)
    rewards = [get_reward_Score.get_score(p, r) for p, r in zip(prompt, responses)]

    ppo_trainer.step(prompt, responses, rewards)





