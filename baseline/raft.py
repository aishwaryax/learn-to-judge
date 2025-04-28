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
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import random

os.environ["HUGGINGFACE_TOKEN"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA.")
    parser.add_argument('--model_repo', type=str, required=True, help="Path to the pre-trained model repository.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the fine-tuned model.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=5e-6, help="Learning rate.")
    parser.add_argument('--reg_lambda', type=float, default=1e-4, help="Regularization lambda value.")
    return parser.parse_args()

args = parse_args()


def get_dataset_specific_args(dataset_path):
    helpsteer_rubrics = """
        [Helpfulness can be measured by how useful and helpful the overall response is.
        While giving score, you can refer the following scoring rubrics. You can only give a single value for the resulting score.]
        Score of 0: The response is not useful or helpful at all. The response completely missed the essence of what the user wanted.
        Score of 1: The response is borderline unhelpful and mostly does not capture what the user was looking for, but is still usable and helpful in a small way.
        Score of 2: The response is partially helpful but misses the overall goal of the user's query/input in some way. The response did not fully satisfy what the user was looking for.
        Score of 3: The response is mostly helpful and mainly aligned with what the user was looking for, but there is still some room for improvement.
        Score of 4: The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for
        """.strip()
    helpsteer_min = 0
    helpsteer_max = 4
    summarize_from_feedback_rubrics = """
        [How good is the summary overall at representing the post? If it's hard to find ways to make the summary better, give the summary a high score. If there are lots of different ways the summary can be made better, give the summary a low score. 
        Judge on the following criteria while giving the feedback:

        Essence: is the summary a good representation of the post?,
        Clarity: is the summary reader-friendly? Does it express ideas clearly?
        Accuracy: does the summary contain the same information as the longer post?
        Purpose: does the summary serve the same purpose as the original post?
        Concise: is the summary short and to-the-point?
        Style: is the summary written in the same style as the original post?

        While giving score, you can refer the following scoring rubrics. Try to interpolate to scores of 2, 3, 5 and 6 as those are not mentioned. You can only give a single value for overall score.
        Score of 1: The summary is terrible.
        Score of 4: The summary is an okay representation of the post, but could be significantly improved.
        Score of 7: The summary is an excellent representation of the post.
        """   
    summarize_from_feedback_min_score = 1 
    summarize_from_feedback_max_score = 7
    if "helpsteer" in dataset_path:
        return (helpsteer_rubrics, helpsteer_min, helpsteer_max)
    elif "summarize_from_feedback" in dataset_path:
        return (summarize_from_feedback_rubrics, summarize_from_feedback_min_score, summarize_from_feedback_max_score)
    else:
        return ()

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

score_token_ids = []
def _create_hf_dataset(data_csv, min_score=1, max_score=5):
    rubrics, min_score, max_score = get_dataset_specific_args(data_csv)
    data_df = pd.read_csv(data_csv)
    data_df.dropna(subset=["human_score", "llm_score"], inplace=True)
    data_df['score'] = pd.to_numeric(data_df['human_score'], errors='coerce')
    score_values = sorted(data_df['score'].dropna().unique())
    score_tokens = [str(int(v)) for v in score_values]
    score_token_ids.extend(tokenizer.convert_tokens_to_ids(score_tokens))
    #short dataset for test
    # data_df = data_df.sample(100, random_state=42)

    def format_prompt(row):
        prompt =  ABSOLUTE_PROMPT_TEMPLATE.format(
            instruction=row['instruction'],
            response=row['response'],
            rubrics=rubrics,
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

class RAFTTrainer(Trainer):
    def __init__(self, *args, score_token_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_token_ids = score_token_ids
        
    def find_marker_positions(self, input_ids: torch.Tensor, marker_ids: list[int]) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        marker_len = len(marker_ids)
        marker_ids = torch.tensor(marker_ids, device=input_ids.device)

        if seq_len < marker_len:
            return torch.full((batch_size,), -1, device=input_ids.device, dtype=torch.long)

        windows = input_ids.unfold(dimension=1, size=marker_len, step=1)
        matches = (windows == marker_ids).all(dim=-1)

        has_match = matches.any(dim=1)
        first_match_idx = matches.float().masked_fill(~matches, float('inf')).argmin(dim=1)

        marker_end_positions = torch.where(
            has_match,
            first_match_idx + marker_len,
            torch.full_like(first_match_idx, -1)
        )
        return marker_end_positions

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")

        marker = "###Score: "
        marker_ids = self.tokenizer.encode(marker, add_special_tokens=False)[1:]  # skip BOS
        marker_len = len(marker_ids)

        batch_size = input_ids.size(0)
        marker_end_positions = self.find_marker_positions(input_ids, marker_ids)
        score_logits = []
        valid_labels = []
        valid = (marker_end_positions != -1) & (marker_end_positions < logits.size(1))
        valid_indices = valid.nonzero(as_tuple=True)[0]

        pos = marker_end_positions[valid_indices]
        batch_indices = valid_indices

        score_logits = logits[batch_indices, pos]
        valid_labels = labels[batch_indices, pos]

        score_logits = score_logits[:, self.score_token_ids]
        probs = F.softmax(score_logits, dim=-1)

        score_values = torch.tensor(
            [int(self.tokenizer.decode(tid)) for tid in self.score_token_ids],
            device=probs.device,
            dtype=torch.float,
        )
        expected_score = (probs * score_values).sum(dim=-1)

        ground_truth = torch.tensor(
            [int(self.tokenizer.decode(lid)) for lid in valid_labels],
            device=probs.device,
            dtype=torch.float,
        )

        loss = F.mse_loss(expected_score, ground_truth.float())

        return (loss, outputs) if return_outputs else loss


trainer = RAFTTrainer(
    model=model,
    args=hf_args,
    train_dataset=tokenized,
    data_collator=collator,
    tokenizer=tokenizer,
    score_token_ids=score_token_ids
)

trainer.train(resume_from_checkpoint=False)

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