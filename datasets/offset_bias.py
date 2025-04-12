import sys
import os
import argparse
from datasets import load_dataset
from baseline.absolute_llm_judge_v3 import AbsoluteLLMJudge
from baseline.relative_llm_judge_v2 import RelativeLLMJudge

dataset = load_dataset("aishwaryaxsahoo/offsetbias")
# dataset["test"] = dataset["test"].select(range(10))
# dataset["train"] = dataset["train"].select(range(10))

parser = argparse.ArgumentParser(description="Run Absolute LLM Judge on Helpsteer Dataset")
parser.add_argument("--dataset_fold", type=str, default="test", help="Dataset split to use (e.g., test, validation)")
parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file")
parser.add_argument("--model_repo", type=str, required=True, help="Path to the model repository")
parser.add_argument("--baseline_type", type=str, required=True, help="Baseline type: absolute or relative")
args = parser.parse_args()

rubrics = """
Does the model provide relevant and useful responses to the user's needs or questions?
"""

one_shot_absolute = """
##Input: 
You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.\n\n###Task Description:\nAn instruction (might include an Input inside it), a response to evaluate, and a score rubric representing an evaluation criteria are given.\n1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.\n2. After writing feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.\n3. The output format should look as follows: "(write feedback for criteria) [RESULT] (an integer number between 1 and 5)"\n4. Please do not generate any other opening, closing, or explanations.\n\n###The instruction to evaluate:\nGenerate a story that explains why the following two facts are true - a llama and a lemur have the same number of teeth.\n\n###Response to evaluate:\nIn the animal kingdom, interesting discoveries often emerge from the most unexpected places. One such discovery is the peculiar similarity between a llama and a lemur in terms of dental arrangement. It all began when Dr. Fluffy Paws, a well-known veterinarian, and amateur anthropologist, was on a field trip in the Andean highlands. While examining a particularly friendly llama, he noted down the llama's dental formula to compare with other species for his research on herbivore dentition.\n\nLater that year, during an expedition in the lush forests of Madagascar, Dr. Paws had the opportunity to study lemurs up close. Remembering his encounter with the llama, he was astonished to find that the dental formula he noted from the lemur seemed vaguely familiar. Upon returning to his lab, Dr. Paws compared his notes and found what appeared to be a remarkable coincidence: the number of teeth in the lemur's mouth was the same as in the llama's mouth he had examined months before. \n\nBaffled by this finding, Dr. Paws constructed a hypothetical narrative pondering if, at some point in the distant past, these two distinct animals could have shared a common, albeit very ancient, ancestor, leading to this unexpected dental similarity. This revelation was shared among the scientific community, sparking discussions and further research into the convergent evolution of species with entirely different lifestyles and habitats. It goes to show that sometimes nature has a humorous way of aligning the stars, or in this case, the teeth, of creatures as disparate as the llama and the lemur.\n\n###Score Rubrics:\n\nDoes the model provide relevant and useful responses to the user's needs or questions?\n\n\n###Feedback:
##Output: The response provides a creative and engaging story that explains why a llama and a lemur have the same number of teeth. It effectively constructs a narrative involving Dr. Fluffy Paws, a fictional veterinarian, to explore the connection between the two animals. The story is relevant to the instruction, as it directly addresses the user's request to explain the shared dental count through a hypothetical scenario. Additionally, the response incorporates scientific curiosity and a plausible explanation (convergent evolution or a distant common ancestor), which adds depth to the narrative.
However, while the response is engaging and imaginative, it does not provide concrete or factual evidence to support the claim. The explanation relies heavily on a fictional narrative rather than offering a more scientific or factual basis for the similarity in the number of teeth. This limits its usefulness for users seeking a more factual or educational response. The response could have been improved by including some factual context about the dental structures of llamas and lemurs or referencing real-world biological principles.
Overall, the response is relevant and creative but lacks factual depth, which slightly diminishes its utility.
[RESULT] 4

### Now your turn:
"""

one_shot_relative = """
##Input:  You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort.\n\n###Task Description:\n An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), and an evaluation criteria are given.\n 1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.\n 2. Make comparisons between Response A and Response B. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.\n 3. After writing the feedback, indicate the better response, either "A" or "B".\n 4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"\n 5. Please do not generate any other opening, closing, and explanations.\n\n ###Instruction:\n Generate a creative Halloween costume idea.\n\n ###Response A:\n A creative Halloween costume idea would be to dress up as a mad scientist, complete with a white lab coat and wild, flyaway hair. Accessories could include (fake) test tubes, jars filled with mysterious liquids and ingredients, and beakers filled with bubbling potions.\n\n ###Response B:\n A creative Halloween costume idea is to dress up as a character from a popular movie or TV show. You can choose a character that you like and try to recreate their iconic look. This can be done by finding clothing items and accessories that resemble what the character wears in the movie or show.\n\nAnother option is to create a costume based on a pun or play on words. For example, you could dress up as a "cereal killer" by wearing a costume that looks like a cereal box with fake blood splatters on it. This type of costume is sure to get some laughs and be a hit at any Halloween party.\n\n ###Score Rubric:\n \nDoes the model provide relevant and useful responses to the user's needs or questions?\n\n\n ###Feedback:
##Output: Both responses address the task of generating a creative Halloween costume idea and provide relevant suggestions. However, there are clear differences in the level of creativity, specificity, and variety offered in the responses, which affects their alignment with the evaluation criteria.
Response A presents a single, specific idea: dressing up as a mad scientist. While this idea is coherent and includes some creative details like using test tubes, jars, and bubbling potions as accessories, it remains relatively conventional and lacks originality. The suggestion does not explore broader or more innovative concepts, focusing on a single, commonly seen costume.
Response B, on the other hand, offers multiple creative approaches. It suggests dressing up as a character from a popular movie or TV show, which is versatile but somewhat generic. However, the inclusion of a second, more imaginative option—a pun-based costume like a "cereal killer"—demonstrates a higher level of creativity and humor. This variety provides the user with multiple ideas to choose from and caters to different tastes, making Response B more engaging and useful.
In terms of the rubric, Response B better fulfills the criterion of providing a relevant and useful response because it offers greater variety and more creative options compared to Response A. While both responses are relevant, Response B stands out for its broader scope and inventiveness.
[RESULT] B

### Now your turn:
"""

def transform_data(example):
    return {
        "instruction": f"{example["instruction"]}",
        "response1": example["output_1"],
        "response2": example["output_2"],
        "human_score": example["label"]-1
    }

offset_bias_transformed_dataset = dataset[args.dataset_fold].map(transform_data)

if args.baseline_type == "absolute":
    absolute_llm_judge = AbsoluteLLMJudge(
        dataset=offset_bias_transformed_dataset, 
        rubrics=rubrics, 
        one_shot=one_shot_absolute,
        output_file=args.output_file, 
        repo_name=args.model_repo, 
        min_score=1, 
        max_score=5
    )
    absolute_llm_judge.generate_inference_file_pair()
else:
    relative_llm_judge = RelativeLLMJudge(
        dataset=offset_bias_transformed_dataset, 
        rubrics=rubrics, 
        one_shot=one_shot_relative,
        output_file=args.output_file, 
        repo_name=args.model_repo
    )
    relative_llm_judge.generate_inference_file()
    


