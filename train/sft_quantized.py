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
warnings.filterwarnings("ignore")


os.environ["HUGGINGFACE_TOKEN"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA.")
    parser.add_argument('--model_repo', type=str, required=True, help="Path to the pre-trained model repository.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the fine-tuned model.")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=5e-6, help="Learning rate.")
    parser.add_argument('--reg_lambda', type=float, default=1e-4, help="Regularization lambda value.")
    return parser.parse_args()

args = parse_args()

rubrics = """
[Helpfulness can be measured by how useful and helpful the overall response is.
While giving score, you can refer the following scoring rubrics. You can only give a single value for the resulting score.]
Score of 0: The response is not useful or helpful at all. The response completely missed the essence of what the user wanted.
Score of 1: The response is borderline unhelpful and mostly does not capture what the user was looking for, but is still usable and helpful in a small way.
Score of 2: The response is partially helpful but misses the overall goal of the user's query/input in some way. The response did not fully satisfy what the user was looking for.
Score of 3: The response is mostly helpful and mainly aligned with what the user was looking for, but there is still some room for improvement.
Score of 4: The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for
""".strip()

ABSOLUTE_PROMPT_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between {min_score} and {max_score}. You should refer to the score rubric.
3. The output format should look as follows: "an integer number between {min_score} and {max_score}"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubrics}

###Score:"""

        
def _create_hf_dataset(data_csv, min_score=1, max_score=5):
    data_df = pd.read_csv(data_csv)
    data_df['score'] = pd.to_numeric(data_df['human_score'], errors='coerce')

    #short dataset for test
    # data_df = data_df.sample(100, random_state=42)

    def format_prompt(row):
        prompt =  ABSOLUTE_PROMPT_TEMPLATE.format(
            instruction=row['instruction'],
            response=row['response'],
            rubrics=rubrics,  # ensure there's a 'rubrics' column in your CSV
            min_score=min_score,
            max_score=max_score,
        )
        return prompt + f" {int(row['human_score'])}" 
    data_df['text'] = data_df.apply(format_prompt, axis=1)
    hf_dataset = Dataset.from_pandas(data_df)
    return hf_dataset

tokenizer = AutoTokenizer.from_pretrained(args.model_repo, use_auth_token=os.environ["HUGGINGFACE_TOKEN"])
tokenizer.pad_token = tokenizer.eos_token
maxseq = 4096

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
marker = "###Score: "
marker_ids = tokenizer.encode(marker, add_special_tokens=False)[1:]
collator = DataCollatorForCompletionOnlyLM(marker_ids, tokenizer=tokenizer)

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
print("Trainable params:\n", trainable)
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
    gradient_accumulation_steps=1,
)

trainer = Trainer(
    model=model,
    args=hf_args,
    train_dataset=tokenized,
    data_collator=collator,
    tokenizer=tokenizer,
)

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