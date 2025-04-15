import os
import re
import csv
import torch
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from vllm import LLM, SamplingParams

class AbsoluteLLMJudge:
    def __init__(self, dataset, rubrics, output_file, repo_name, min_score=1, max_score=5, processed_indices={}):
        self.dataset = dataset
        self.rubrics = rubrics
        self.output_file = output_file
        self.min_score = min_score
        self.max_score = max_score
        self._load_model_tokenizer(repo_name)
        self.processed_indices = processed_indices

    def _load_model_tokenizer(self, repo_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = LLM(model=repo_name, tensor_parallel_size=torch.cuda.device_count())
        self.tokenizer = AutoTokenizer.from_pretrained(repo_name)

    def _get_judge_llm_resp(self, instruction, response):
        ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
        safe_instruction = instruction.replace("{", "{{").replace("}", "}}")
        safe_response = response.replace("{", "{{").replace("}", "}}") if response else ""

        prompt = f"""###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between {self.min_score} and {self.max_score}. You should refer to the score rubric.
        3. The output format should look as follows: \"(write a feedback for criteria) [RESULT] (an integer number between {self.min_score} and {self.max_score})\"
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        {instruction}

        ###Response to evaluate:
        {response}

        ###Score Rubrics:
        {self.rubrics}

        ###Feedback: """

        user_content = ABS_SYSTEM_PROMPT + "\n\n" + prompt
        sampling_params = SamplingParams(max_tokens=1000, temperature=0.1, top_p=0.9)
        outputs = self.llm.generate([user_content], sampling_params)
        return outputs[0].outputs[0].text.strip(), user_content

    def _parse_feedback_and_score(self, text):
        result_match = re.search(r"\[RESULT\]\s*(\d+)", text)
        if result_match:
            score = int(result_match.group(1))
            feedback = text[:result_match.start()].strip()
        else:
            score = None
            feedback = text.strip()
        return feedback, score

    def _get_processed_lines(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader, None)
                return sum(1 for row in reader)
        else:
            return 0

    def reward_fn(self, pred_score, true_score):
        return 1 if pred_score == true_score else 0

    def generate_inference_file(self, batch_size=100):
        start_idx = self._get_processed_lines()
        output_dir = Path(self.output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        with open(self.output_file, mode='a+', newline='') as file:
            writer = csv.writer(file)
            if os.stat(self.output_file).st_size == 0:
                writer.writerow(["instruction", "response", "llm_prompt", "human_score", "llm_score", "llm_critique", "llm_response"])

        for idx, item in enumerate(self.dataset):
            if idx < start_idx:
                continue
            instruction = item["instruction"]
            if instruction is None:
                continue
            response = item["response"]
            human_score = item["human_score"]

            llm_response, llm_prompt = self._get_judge_llm_resp(instruction, response)
            feedback, llm_score = self._parse_feedback_and_score(llm_response)
            results.append([instruction, response, llm_prompt, human_score, llm_score, feedback, llm_response])

            if len(results) >= batch_size:
                self._write_batch(results)
                self.prepare_sft_data("sft_batch.jsonl")
                self.fine_tune_model("sft_batch.jsonl", output_dir=f"judge-ft-{idx}")
                results = []

        if results:
            self._write_batch(results)
            self.prepare_sft_data("sft_batch.jsonl")
            self.fine_tune_model("sft_batch.jsonl", output_dir=f"judge-ft-final")

    def _write_batch(self, results):
        with open(self.output_file, mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)

    def prepare_sft_data(self, sft_file):
        df = pd.read_csv(self.output_file)
        examples = []

        for _, row in df.iterrows():
            try:
                pred_score = float(row["llm_score"])
                human_score = float(row["human_score"])
            except:
                continue

            reward = self.reward_fn(pred_score, human_score)
            if reward == 1:
                prompt = f"""###Task Description:
                An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
                1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
                2. After writing a feedback, write a score that is an integer between {self.min_score} and {self.max_score}. You should refer to the score rubric.
                3. The output format should look as follows: \"(write a feedback for criteria) [RESULT] (an integer number between {self.min_score} and {self.max_score})\"
                4. Please do not generate any other opening, closing, and explanations.

                ###The instruction to evaluate:
                {row['instruction']}

                ###Response to evaluate:
                {row['response']}

                ###Score Rubrics:
                {self.rubrics}

                ###Feedback: """
                target = row["llm_response"].strip()
                examples.append({"text": prompt + target})

        if not examples:
            print("⚠️ No high-reward samples found.")
            return

        pd.DataFrame(examples).to_json(sft_file, orient="records", lines=True)
        print(f"✅ Saved {len(examples)} SFT examples to {sft_file}")

    def fine_tune_model(self, sft_file, output_dir="judge-ft", epochs=2):
        dataset = load_dataset("json", data_files={"train": sft_file})["train"]

        def tokenize(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

        tokenized = dataset.map(tokenize, batched=True)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            num_train_epochs=epochs,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            logging_dir=os.path.join(output_dir, "logs")
        )

        trainer = Trainer(
            model=self.llm.llm_engine.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=tokenized
        )

        trainer.train()
