import re
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
import argparse
import gc
from torch.utils.data import Dataset, DataLoader

gc.collect()
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA.")
    parser.add_argument('--model_repo', type=str, required=True, help="Path to the pre-trained model repository.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the fine-tuned model.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate.")
    parser.add_argument('--reg_lambda', type=float, default=1e-4, help="Regularization lambda value.")
    return parser.parse_args()

args = parse_args()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(args.model_repo)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model_repo, quantization_config=bnb_config).to(device)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)

df = pd.read_csv(args.dataset_path)
df = df.dropna(subset=['human_score', 'llm_score'])
dataset = list(df[['llm_prompt', 'llm_response', 'llm_score', 'human_score']].itertuples(index=False, name=None))

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        llm_prompt, llm_response, llm_score, human_score = self.data[idx]
        return llm_prompt, llm_response, int(llm_score), int(human_score)

def load_data(dataset, batch_size, shuffle=True):
    custom_dataset = CustomDataset(dataset)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def compute_reward(s_hat, s):
    return 1.0 if int(s_hat) == int(s) else 0.0

def sft_update_batch(model, tokenizer, batch_data, weights, optimizer):
    losses = []
    max_length = 1000
    for (llm_prompt, llm_response, llm_score, human_score), weight in zip(batch_data, weights):
        inputs = tokenizer(llm_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
        targets = tokenizer(llm_response, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
        outputs = model(**inputs, labels=targets["input_ids"])
        loss = outputs.loss
        weighted_loss = weight * loss
        losses.append(weighted_loss)
    if losses:
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return total_loss.item()
    return 0.0

def train_judge_model(model, tokenizer, dataset, epochs=5, batch_size=16, lr=1e-5, reg_lambda=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_lambda)
    dataloader = load_data(dataset, batch_size)
    
    for epoch in range(epochs):
        model.train()
        total_rewards = 0
        total_loss = 0
        for batch_data in dataloader:
            llm_prompt_batch, llm_response_batch, llm_score_batch, human_score_batch = batch_data
            
            rewards = [compute_reward(llm_score, human_score) for llm_score, human_score in zip(llm_score_batch, human_score_batch)]
            loss = sft_update_batch(model, tokenizer, zip(llm_prompt_batch, llm_response_batch, llm_score_batch, human_score_batch), rewards, optimizer)
            
            total_rewards += sum(rewards)
            total_loss += loss
        
        print(f"Epoch {epoch + 1}: Mean Reward = {total_rewards / len(dataset):.3f}, Loss = {total_loss / len(dataset):.4f}")

train_judge_model(model, tokenizer, dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, reg_lambda=args.reg_lambda)

model.save_pretrained(args.save_path)
tokenizer.save_pretrained(args.save_path)
print(f"Model saved to {args.save_path}")
