from transformers import TrainerState, TrainerControl, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
import pandas as pd
import numpy as np
from datasets import Dataset
import os
import gc
import torch
import pandas as pd
import transformers
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import bitsandbytes as bnb
from trl import DataCollatorForCompletionOnlyLM
import argparse
import json
from torch.utils.data import DataLoader

os.environ["HUGGINGFACE_TOKEN"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

        
def _create_hf_dataset(data_csv):
    data_df = pd.read_csv(data_csv)
    data_df['llm_score'] = pd.to_numeric(data_df['llm_score'], errors='coerce')
    data_df['human_score'] = pd.to_numeric(data_df['human_score'], errors='coerce')
    data_df = data_df.dropna(subset=['llm_score', 'human_score'])
    data_df = data_df[data_df['llm_score'].astype(int) == data_df['human_score'].astype(int)]
    data_df['text'] = data_df['llm_prompt'] + data_df['llm_response']
    hf_dataset = Dataset.from_pandas(data_df)
    return hf_dataset

tokenizer = AutoTokenizer.from_pretrained(args.model_repo, use_auth_token=os.environ["HUGGINGFACE_TOKEN"])
tokenizer.pad_token = tokenizer.eos_token

train_dataset = _create_hf_dataset(args.dataset_path)
maxseq = 4096
print(f"Number of rows in training set before filtering: {len(train_dataset)}")

train_dataset = train_dataset.filter(lambda x: len(tokenizer(x['text'])["input_ids"]) <= maxseq).shuffle(seed=42)
# train_dataset = train_dataset.filter(lambda x: len(tokenizer(x['text'])["input_ids"]) <= maxseq).shuffle(seed=42)
response_template_ids = None
if "prometheus" in args.model_repo:
    response_template_with_context = "[/INST]"
    response_template_ids = tokenizer.encode(response_template_with_context)[1:]
else:
    response_template_with_context = 'assistant<|end_header_id|>'
    response_template_ids = tokenizer.encode(response_template_with_context)[1:]
    
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

print(f"Number of rows in training set: {len(train_dataset)}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_repo,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
)

model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

training_args = SFTConfig(
    output_dir=args.save_path,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    group_by_length=True,
    bf16=False,
    learning_rate=args.lr,
    optim="paged_adamw_32bit",
    logging_strategy='steps',
    logging_steps=128,
    save_steps=2,
    save_strategy='epoch',
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    dataloader_prefetch_factor=1,
    logging_first_step=True,
    lr_scheduler_type="cosine",
    seed=42,
    disable_tqdm=False,
    dataset_text_field="text",
    packing=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=config,
    args=training_args,
    data_collator=collator,
)

trainer.train()

trainer.model.save_pretrained(f"{args.save_path}/final_checkpoint")
tokenizer.save_pretrained(f"{args.save_path}/final_checkpoint")

del trainer, model
gc.collect()
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    args.model_repo,
    return_dict=True,
    torch_dtype=torch.float16,
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
)

base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base_model, model_id=f"{args.save_path}/final_checkpoint", use_auth_token=os.environ["HUGGINGFACE_TOKEN"])
model = model.merge_and_unload()

model.save_pretrained(args.save_path + "/sft")
tokenizer.save_pretrained(args.save_path + "/sft")

del model, base_model
gc.collect()
torch.cuda.empty_cache()