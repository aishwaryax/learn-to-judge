from datasets import load_dataset
import os
import re
import csv
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path

class RelativeLLMJudge:
    def __init__(self, dataset, rubrics, output_file, repo_name, parse_feedback_and_score, processed_indices={}):
        self.dataset = dataset
        self.rubrics = rubrics
        self.output_file = output_file
        self._load_model_tokenizer(repo_name)
        self.processed_indices = processed_indices
        self._parse_feedback_and_score = parse_feedback_and_score
        
    def _load_model_tokenizer(self, repo_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = LLM(model=repo_name, tensor_parallel_size=torch.cuda.device_count())
        self.tokenizer = AutoTokenizer.from_pretrained(repo_name)
        
    def _get_judge_llm_resp(self, prompt):
        sampling_params = SamplingParams(max_tokens=1000, temperature=0.1, top_p=0.9)
        outputs = self.llm.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _get_processed_lines(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader, None)
                return sum(1 for row in reader)
        else:
            return 0
                
    def generate_inference_file(self):
        start_idx = self._get_processed_lines()
        output_dir = Path(self.output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        batch_size = 5

        with open(self.output_file, mode='a+', newline='') as file:
            writer = csv.writer(file)
            if os.stat(self.output_file).st_size == 0:
                writer.writerow(["instruction", "response1", "response2", "llm_prompt", "human_score", "llm_critique", "llm_score", "llm_response"])

        for idx, item in enumerate(self.dataset):
            if idx < start_idx:
                continue
            instruction = item["instruction"]
            if instruction is None:
                continue
            response1 = item["response1"]
            response2 = item["response2"]
            human_score = item["human_score"]
            prompt = item["prompt"]
            llm_response = self._get_judge_llm_resp(prompt)
            print(llm_response)
            feedback, score = self._parse_feedback_and_score(llm_response)
            if score is None:
                continue
            print(score)
            results.append([instruction, response1, response2, prompt, human_score, feedback, score, llm_response])
            if len(results) >= batch_size:
                with open(self.output_file, mode='a+', newline='') as file:        
                    writer = csv.writer(file)
                    writer.writerows(results)
                    results = []

        if results:
            with open(self.output_file, mode='a+', newline='') as file:        
                writer = csv.writer(file)
                writer.writerows(results)