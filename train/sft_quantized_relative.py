from transformers import TrainerState, TrainerControl, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, Trainer
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
import warnings
from transformers.trainer_utils import get_last_checkpoint
from rubric_config import RUBRIC_CONFIG
warnings.filterwarnings("ignore")


os.environ["HUGGINGFACE_TOKEN"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA.")
    parser.add_argument('--model_repo', type=str, required=True, help="Path to the pre-trained model repository.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the fine-tuned model.")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=5e-6, help="Learning rate.")
    parser.add_argument('--reg_lambda', type=float, default=1e-4, help="Regularization lambda value.")
    return parser.parse_args()

args = parse_args()

RELATIVE_PROMPT_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given evaluation criteria,, not evaluating in general.
2. Make comparisons between Response A and Response B. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing the feedback, indicate the better response, either "A" or "B".
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response1}

###Response B:
{response2}

###Score Rubric:
{rubrics}

###Result: """

def get_rubric_and_scores(dataset_path):
    print(f"Dataset path: {dataset_path}")
    if "offset_bias" in dataset_path.lower():
        return RUBRIC_CONFIG["offset_bias"]
    elif "nectar" in dataset_path.lower():
        return RUBRIC_CONFIG["nectar"]
    else:
        raise ValueError("Unknown dataset type in path.")


def _create_hf_dataset(data_csv):
    data_df = pd.read_csv(data_csv)
    data_df = data_df.dropna(subset=["human_score"])
    data_df['score'] = pd.to_numeric(data_df['human_score'], errors='coerce')

    rubric_config = get_rubric_and_scores(data_csv)

    #short dataset for test
    # data_df = data_df.sample(100, random_state=42)

    def format_prompt(row):
        prompt =  RELATIVE_PROMPT_TEMPLATE.format(
            instruction=row['instruction'],
            response1=row['response1'],
            response2=row['response2'],
            rubrics=rubric_config["rubric"],  # ensure there's a 'rubrics' column in your CSV
        )
        # is human score is 0 then A, if 1 then B
        if row['human_score'] == 0:
            prompt += f"A"
        else:
            prompt += f"B"
        return prompt
    data_df['text'] = data_df.apply(format_prompt, axis=1)
    hf_dataset = Dataset.from_pandas(data_df)
    return hf_dataset

tokenizer = AutoTokenizer.from_pretrained(args.model_repo, use_auth_token=os.environ["HUGGINGFACE_TOKEN"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
maxseq = 8192

raw_ds = _create_hf_dataset(args.dataset_path)
raw_ds = raw_ds.filter(lambda x: len(tokenizer(x["text"], add_special_tokens=False)["input_ids"]) <= maxseq)
raw_ds = raw_ds.shuffle(seed=42)
# print(f"Number of rows in training set before filtering: {len(train_dataset)}")


# 3) Pre-tokenize
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=maxseq,
        return_attention_mask=True,
        padding=False,
    )
tokenized = raw_ds.map(tokenize_fn, batched=True, remove_columns=raw_ds.column_names)# train_dataset = train_dataset.filter(lambda x: len(tokenizer(x['text'])["input_ids"]) <= maxseq).shuffle(seed=42)

# 4) Build collator for completion-only loss
marker = "###Result:"
marker_ids = tokenizer.encode(marker, add_special_tokens=False)[1:]
collator = DataCollatorForCompletionOnlyLM(marker_ids, tokenizer=tokenizer)

#debug code
# for idx in range(5):
#     input_ids = tokenized[idx]['input_ids']
#     text = tokenizer.decode(input_ids)
#     print(f"\n=== Example {idx} ===")
#     print("Decoded text:\n", text)

#     marker_text = marker
#     print("Full text token IDs:", input_ids)
#     print("Marker token IDs:", marker_ids)

#     found = False
#     for j in range(len(input_ids) - len(marker_ids) + 1):
#         if input_ids[j:j+len(marker_ids)] == marker_ids:
#             found = True
#             print(f"Marker found at position {j}")
#             break
#     if not found:
#         print("Marker NOT found in this example!")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# 5) Load base model (quantized) and prepare for 4-bit training
model = AutoModelForCausalLM.from_pretrained(
    args.model_repo,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)
model.resize_token_embeddings(len(tokenizer))

# 6) Now attach your LoRA adapters
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.gradient_checkpointing_enable()

for name, param in model.named_parameters():
    # PEFT adapter weights have "lora_" in their names by default
    if "lora_" not in name:
        param.requires_grad = False

trainable = [n for n,p in model.named_parameters() if p.requires_grad]
# print("Trainable params:\n", trainable)
print(f"\nTotal trainable params: {sum(param.numel() for _, param in model.named_parameters() if param.requires_grad)}")

# 7) Configure & launch Trainer
hf_args = TrainingArguments(
    output_dir=args.save_path,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.lr,
    logging_steps=32,
    save_strategy="epoch",
    save_steps=1,
    fp16=True,
    gradient_accumulation_steps = 1,    
)

trainer = Trainer(
    model=model,
    args=hf_args,
    train_dataset=tokenized,
    data_collator=collator,
    tokenizer=tokenizer,
)


# last_ckpt = get_last_checkpoint(args.save_path)  # e.g. ".../checkpoint-3000"
# if last_ckpt is None:
#     raise ValueError(f"No checkpoint found in {args.save_path}")
# print(f"➡️  Resuming from {last_ckpt}")

trainer.train()

# 8) Merge adapters & save
model.save_pretrained(f"{args.save_path}/final_checkpoint")
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