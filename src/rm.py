import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name)
tokenizer = AutoTokenizer.from_pretrained(reward_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rank_model.to(device)
rank_model.eval()


def get_reward_score(question, answer):
    text = f"{question}\n{answer}"

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        score = rank_model(**inputs).logits[0].item()  

    return score
