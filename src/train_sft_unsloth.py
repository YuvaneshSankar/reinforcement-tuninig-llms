import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
#we can use unsloth import UnslothSFTTrainer for better integration, but here we use trl's SFTTrainer as an example
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq=2048

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name="unsloth/tinyllama-bnb-4bit",
  max_seq_length=max_seq,
  dtype=None,  # Auto-detects best dtype
  load_in_4bit=True,  # QLoRA
)

target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

model=FastLanguageModel.get_peft_model(
  model,
  r=16,
  target_modules=target_modules,
  lora_alpha=16,
  lora_dropout=0,  # Unsloth recommends 0
  bias="none",
  task_type="CAUSAL_LM"
)

dataset = load_dataset("tatsu-lab/alpaca", split="train")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}"
        if input_text:
            text += f"\n### Input:\n{input_text}"
        text += f"\n### Response:\n{output}"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

trainer= SFTTrainer(
  model=model,
  tokenizer=tokenizer,
  train_dataset=dataset,
  dataset_text_field="text",
  max_seq_length=max_seq,
  packing=True,  # Packs sequences for efficiency
  args=TrainingArguments(
    output_dir="./models/sft_unsloth",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=1000, #each step processes one effective batch (per_device_train_batch_size * gradient_accumulation_steps = 4*4=16 samples here). 1000 steps ≈ 1000 forward/backward passes on ~16k samples from Alpaca subset (~52k total train split)
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    seed=3407,
  ),
)

trainer.train()
model.save_pretrained("./models/sft_unsloth")