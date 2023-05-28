import time
import pandas as pd
import os
from src.utils import Prompt
import json
from src.utils import LLMModel


class EntailmentFeedback(Prompt):
    def __init__(
        self,
        engine: str,
        prompt_examples: str,
        temperature: float,
        max_tokens: int = 300,
    ) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n### END ###n\n",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.instruction = "# There is an error in above entailment tree above because of lack of understanding of the problem. What is the error? To find the error, go through the entailment tree line by line, and check if everything looks good."
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, prompt_examples: str) -> str:
        if prompt_examples is None:
            return None
        elif not os.path.exists(prompt_examples):
            raise FileNotFoundError(
                f"Prompt examples file {prompt_examples} not found."
            )
        elif prompt_examples.endswith(".json"):
            with open(prompt_examples, "r") as f:
                examples = json.load(f)
            self.prompt = "\n\n".join(
                [
                    f"{self.question_prefix}{example['question']}{self.intra_example_sep}{self.answer_prefix}{example['solution']}"
                    for example in examples
                ]
            )
        else:
            with open(prompt_examples, "r") as f:
                self.prompt = f.read()

    def __call__(self, solution: str):
        generation_query = self.make_query(solution=solution)
        # print(generation_query)
        llm = LLMModel(
            engine=self.engine,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop_str="### END",
        )
        success = False
        while not success:
            try:
                output = llm([generation_query])
                success = True
            except Exception as e:
                success = False
                print(e)
                time.sleep(60)

        if "### END" in output:
            entire_output = output.split("### END")[0]
        feedback = output.strip()
        return feedback

    def make_query(self, solution: str):
        solution = f"""{solution}{self.intra_example_sep}{self.instruction}"""
        return f"{self.prompt}{solution}"


def test():
    task_fb = GSMFeedback(
        prompt_examples="data/prompt/gsm_iter/feedback.txt",
        engine="text-davinci-002",
        temperature=0.7,
    )

    wrong_soln = """def solution():
    \"\"\"Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.\"\"\"
    plates = 6
    plate_cost = 6000
    cups = 12 * 20
    cup_cost = (plates * plate_cost) / cups - 1200
    result = cup_cost
    return result"""
    feedback = task_fb(wrong_soln)
    print(f"Feedback: {feedback}")


if __name__ == "__main__":
    test()
