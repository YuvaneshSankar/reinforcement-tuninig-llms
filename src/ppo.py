import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer
from trl import PPOTrainer , PPOConfig
from unsloth import FastLanguageModel
from peft import PeftModel
from torch.utils.data import DataLoader
from datasets import load_dataset
from trl import AutoModelForCausalLMWithValueHead



dataset = load_dataset("tatsu-lab/alpaca", split="train")
prompts = dataset['instruction']


dataloader = DataLoader(
    prompts,
    batch_size=16,
    shuffle=True,
    collate_fn=lambda x: x
)



def get_reward_score(question, answer):
    reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name)
    tokenizer = AutoTokenizer.from_pretrained(reward_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank_model.to(device)
    rank_model.eval()
    text = f"{question}\n{answer}"

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        score = rank_model(**inputs).logits[0].item()

    return score

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


def ppo_training():
    base_model, tokenizer = load_model_and_tokenizer(2048, "unsloth/tinyllama-bnb-4bit")

    #create value modle before laoding adapter
    peft_model = load_adapter(base_model, "./models/sft/checkpoint-6000")
    model=peft_model
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)

    train_dataset = dataset.rename_column("instruction", "query")

    #Dont need to do this by deafualt if ref_model in ppo trainer is not passed it takes the policy model and creates a copy for reference model

    # ref_model = load_adapter(base_model, "./models/sft/checkpoint-6000")
    # ref_model.eval()
    # for param in ref_model.parameters():
    #     param.requires_grad = False

    ppo_config = PPOConfig(
        kl_coef=0.05,
        gamma=1, # discount factor
        lam=0.95, # GAE lambda
        cliprange_value=0.2, # clipping range for value function
        vf_coef=0.1, # value function coefficient
    )

    # ppo_trainer = PPOTrainer(
    #     args=ppo_config,
    #     model=model,
    #     ref_model=ref_model,
    #     tokenizer=tokenizer
    # )

    ppo_trainer = PPOTrainer(
        config=ppo_config,  # Use 'config' instead of 'args'
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        value_model=value_model,
        train_dataset=train_dataset,
    )

    for epoch in range(3):
        for batch in dataloader:

            query_tensors = [tokenizer(p, return_tensors="pt").input_ids.squeeze(0) for p in batch]

            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=128,
                do_sample=True, #we will use sampling for more diverse outputs
                temperature=0.7,
                top_p=0.9
            )

            # Decode responses
            responses_text = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            rewards = [torch.tensor(get_reward_score(p, r)) for p, r in zip(batch, responses_text)]

            # PPO update
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            # Log every 10 steps
            if ppo_trainer.step_count % 10 == 0:
                print(f"Epoch {epoch}, Step {ppo_trainer.step_count}, Reward: {torch.stack(rewards).mean():.2f}")


        model.save_pretrained(f"./models/ppo/epoch_{epoch}")



if __name__ == "__main__":
  ppo_training()
