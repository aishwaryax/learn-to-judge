from datasets import load_dataset
import os
import re
import csv
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path

class RelativeLLMJudge:
    def __init__(self, dataset, rubrics, one_shot, output_file, repo_name, processed_indices={}):
        self.dataset = dataset
        self.rubrics = rubrics
        self.one_shot = one_shot
        self.output_file = output_file
        self._load_model_tokenizer(repo_name)
        self.processed_indices = processed_indices
        
        
    def _load_model_tokenizer(self, repo_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = LLM(model=repo_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.9)
        self.tokenizer = AutoTokenizer.from_pretrained(repo_name)
        
    def _get_judge_llm_resp(self, instruction, response1, response2):
        REL_SYSTEM_PROMPT = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."
        safe_instruction = instruction.replace("{", "{{").replace("}", "}}")
        safe_response1 = response1.replace("{", "{{").replace("}", "}}")
        safe_response2 = response2.replace("{", "{{").replace("}", "}}")

        RELATIVE_PROMPT_WO_REF = f"""###Task Description:
        An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), and an evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.
        2. Make comparisons between Response A and Response B. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
        3. After writing the feedback, indicate the better response, either "A" or "B".
        4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
        5. Please do not generate any other opening, closing, and explanations.

        ### Example: 
        {self.one_shot}

        ###Instruction:
        {safe_instruction}

        ###Response A:
        {safe_response1}

        ###Response B:
        {safe_response2}

        ###Score Rubric:
        {self.rubrics}

        ###Feedback: """
        user_content = REL_SYSTEM_PROMPT + "\n\n" + RELATIVE_PROMPT_WO_REF
        sampling_params = SamplingParams(max_tokens=1000, temperature=0.1, top_p=0.9)
        outputs = self.llm.generate([user_content], sampling_params)
        return outputs[0].outputs[0].text.strip(), user_content

    def _parse_feedback_and_score(self, text):
        result_match = re.search(r"\[RESULT\]\s*([AB])", text)
        if result_match:
            score = result_match.group(1)
            feedback = text[:result_match.start()].strip()
        else:
            score = None
            feedback = text.strip()
        if score in ['A', 'B']:
            score = 0 if score == 'A' else 1
        else:
            score = None
        return feedback, score
    
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
        batch_size = 100

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
            llm_response, llm_prompt = self._get_judge_llm_resp(instruction, response1, response2)
            feedback, score = self._parse_feedback_and_score(llm_response)
            if score is None:
                continue
            results.append([instruction, response1, response2, llm_prompt, human_score, feedback, score, llm_response])
            if len(results) >= batch_size:
                with open(self.output_file, mode='a+', newline='') as file:        
                    writer = csv.writer(file)
                    writer.writerows(results)
                    results = []

        if results:
            with open(self.output_file, mode='a+', newline='') as file:        
                writer = csv.writer(file)
                writer.writerows(results)
