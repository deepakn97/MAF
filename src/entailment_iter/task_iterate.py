import sys
import time
from typing import Dict, List
from src.utils import Prompt, LLMModel


class EntailmentIterate(Prompt):
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
            inter_example_sep="\n\n",
        )
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.instruction = "# Given the feedback and the original code, let's rewrite the code to incorporate all of the feedback."
        self.setup_prompt_from_examples_file(prompt_examples)

    # do we actually have prompt examples for feedback + model iteration?
    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def __call__(self, solution: str, feedback: str) -> str:
        generation_query = self.make_query(solution=solution, feedback=feedback)
        # print(generation_query)
        success = False
        llm = (
            LLMModel(
                engine=self.engine,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                instruction=self.instruction,
                question_prefix=self.question_prefix,
                answer_prefix=self.answer_prefix,
                intra_example_sep=self.intra_example_sep,
                inter_example_sep=self.inter_example_sep,
            ),
        )
        while not success:
            try:
                output = llm([generation_query])[0]
                success = True
            except Exception as e:
                success = False
                print(e)
                time.sleep(60)

        # print(f"Iterate Output: {entire_output}")
        if "### END ###" in output:
            output = output.split("### END ###")[0].strip()
        return solution

    def make_query(self, solution: str, feedback: str) -> str:
        solution = f"""{solution}{self.intra_example_sep}Feedback:\n{feedback}{self.intra_example_sep}{self.instruction}"""
        query = f"{self.prompt}{self.intra_example_sep}{solution}"
        return query


def test():
    task_iterate = GSMIterate(
        engine="text-davinci-002",
        prompt_examples="data/prompt/gsm_iter/iterate.txt",
        temperature=0.7,
    )

    wrong_soln = """def solution():
  \"\"\"Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.\"\"\"
    plates = 6
    plate_cost = 6000
    cups = 12 * 20
    cup_cost = plate_cost
    result = cup_cost
    return result"""
    feedback = """# Let us go through the error and check step-by-step
    plates = 6
    plate_cost = 6000
# looks good

# Let's check the other parts
    cups = 12 * 20
    cup_cost = plate_cost
# wrong! The cost of a cup is not the same as the cost of a plate. The cost of a cup is $1200 less than the total cost of half a dozen plates sold at $6000 each. So we need to calculate the cost of a cup first (total cost of half a dozen plates sold at $6000 each - $1200) and use that."""
    print(task_iterate(wrong_soln, feedback))


if __name__ == "__main__":
    test()
