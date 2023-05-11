import sys
import time
from typing import Dict

from src.utils import Prompt
import pandas as pd
from pathlib import Path
from prompt_lib.backends import openai_api

from src.utils import Prompt, LLMFeedback, FeedbackFactory



@FeedbackFactory.register("missing_step")
class MissingStepFeedback(LLMFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Missing Step Feedback", max_tokens=300, answer_prefix="def solution():", **kwargs)
        self.instruction = """# Check each semantically complete block of code for any missing steps and suggest the correct way to add them. Ignore all the other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)

@FeedbackFactory.register("variable_naming")
class VariableNameFeedback(LLMFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs
    ) -> None:
        super().__init__(name="Variable Naming Feedback", max_tokens=600, eager_refine=True, answer_prefix="def solution():", **kwargs)
        self.instruction = """# Check each semantically complete block of code and identify the variables that are not named correctly or may cause confusion and fix the issues. State the assumptions you made when renaming the variables clearly. Ignore all the other type of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)

@FeedbackFactory.register("logical")
class LogicalFeedback(LLMFeedback):
    def __init__(
        self,
        prompt_examples: str,
        **kwargs,
    ) -> None:
        super().__init__(name="Logical Reasoning Feedback", max_tokens=300, answer_prefix="def solution():", **kwargs)
        self.instruction = """# Check each semantically complete block of the code to check for any logical reasoning errors. Logical reasoning errors may include errors in the mathematical calculations, errors in the order of the steps, or errors in the assumptions made. State the assumptions you made clearly. Ignore all the other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)
        
class HallucinationFeedback(Prompt):
    def __init__(
        self,
        engine: str,
        prompt_examples: str,
        temperature: float,
        max_tokens: int = 600,
    ) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n### END ###\n\n",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.instruction = """# Check each semantically complete block of code for any hallucination errors and suggest fixes. Hallucination errors are steps that are supported by neither the context nor the real world. Ignore all other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def make_query(self, solution: str):
        solution = f"""{solution}{self.intra_example_sep}{self.instruction}"""
        return f"{self.prompt}{solution}"

    def __call__(self, solution: str) -> Dict[str, str]:
        generation_query = self.make_query(solution=solution)
        success = False
        while not success:
            try:
                output = openai_api.OpenaiAPIWrapper.call(
                    prompt=generation_query,
                    engine=self.engine,
                    max_tokens=self.max_tokens,
                    stop_token="### END",
                    temperature=self.temperature,
                )
                success = True
            except Exception as e:
                success = False
                print(e)
                time.sleep(60)

        entire_output = openai_api.OpenaiAPIWrapper.get_first_response(output)
        if "### END" in entire_output:
            entire_output = entire_output.split("### END")[0]
        fb_and_maybe_soln = entire_output.strip()
        if "def solution():" in fb_and_maybe_soln:
            feedback = fb_and_maybe_soln.split('def solution():')[0].strip()
            solution = fb_and_maybe_soln.split('def solution():')[1].rstrip()
            solution = f"def solution():{solution}"
        else:
            feedback = fb_and_maybe_soln
            solution = ""

        return {"feedback": feedback, "solution": solution}


class CoherencyFeedback(Prompt):
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
            inter_example_sep="\n\n### END ###\n\n",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.instruction = """# Check the code for any coherency errors and suggest fixes. Coherency errors are steps that contradict each other or do not follow a cohesive story. Ignore all other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def make_query(self, solution: str):
        solution = f"""{solution}{self.intra_example_sep}{self.instruction}"""
        return f"{self.prompt}{solution}"

    def __call__(self, solution: str):
        generation_query = self.make_query(solution=solution)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="### END",
            temperature=self.temperature,
        )

        entire_output = openai_api.OpenaiAPIWrapper.get_first_response(output)
        if "### END" in entire_output:
            entire_output = entire_output.split("### END")[0]
        feedback = entire_output.strip()
        return feedback


def test():
    # missing_step = MissingStepFeedback(
    #     prompt_examples="prompt/gsm_maf/missing_step.txt",
    #     engine="text-davinci-003",
    #     temperature=0.7
    # )
    # variable_naming = VariableNameFeedback(
    #     prompt_examples="prompt/gsm_maf/variable_naming.txt",
    #     engine="text-davinci-003",
    #     temperature=0.7,
    # )
    # logical = LogicalFeedback(
    #     prompt_examples="prompt/gsm_maf/logical.txt",
    #     engine="text-davinci-003",
    #     temperature=0.7,
    # )
    print(FeedbackFactory.registry)
    missing_step = FeedbackFactory.create_feedback('missing_step', engine='text-davinci-003', temperature=0.7, prompt_examples='prompt/gsm_maf/missing_step.txt', answer_prefix="def solution():")
    variable_naming = FeedbackFactory.create_feedback('variable_naming', engine='text-davinci-003', temperature=0.7, prompt_examples='prompt/gsm_maf/variable_naming.txt', answer_prefix='def solution():', max_tokens=600)
    logical = FeedbackFactory.create_feedback('logical', engine='text-davinci-003', temperature=0.7, prompt_examples='prompt/gsm_maf/logical.txt', answer_prefix='def solution():')

    wrong_solns = ["""def solution():
    \"\"\"Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?\"\"\"
    chips_per_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
    chips_needed = height * chips_per_inch
    chips_available = bags * chips_per_bag
    chips_left = chips_available - chips_needed
    length = chips_left / chips_per_inch
    result = length
    return result""",
    """def solution():
    \"\"\"Kelly is grocery shopping at a supermarket and is making sure she has enough in her budget for the items in her cart. Her 5 packs of bacon cost $10 in total and she has 6 packets of chicken which each cost twice as much as a pack of bacon. She also has 3 packs of strawberries, priced at $4 each, and 7 packs of apples, each priced at half the price of a pack of strawberries. If Kelly’s budget is $65 then how much money, in dollars, does she have left in her budget?\"\"\"
    budget = 65
    bacon_packs = 5
    bacon_total_cost = 10
    chicken_packs = 6
    chicken_cost = 2 * bacon_cost
    strawberry_packs = 3
    strawberry_cost = 4
    apple_packs = 7
    apple_cost = strawberry_cost / 2
    total_cost = bacon_cost + chicken_cost + strawberry_cost + apple_cost
    money_left = budget - total_cost
    result = money_left
    return result"""
    ]
    vn_feedback_and_solns = variable_naming(wrong_solns)
    for i, vn_feedback_and_soln in enumerate(vn_feedback_and_solns):
        print(f"Variable Naming Feedback {i}:\n{vn_feedback_and_soln['feedback']}")
        print(f"Variable Naming Solution {i}:\n{vn_feedback_and_soln['solution']}")


    ms_feedbacks = missing_step([x['solution'] for x in vn_feedback_and_solns])
    print(len(ms_feedbacks))
    for i, ms_feedback in enumerate(ms_feedbacks):
        print(f"Missing Step Feedback {i}:\n{ms_feedback}")

    logical_feedbacks = logical([x['solution'] for x in vn_feedback_and_solns])
    for i, logical_feedback in enumerate(logical_feedbacks):
        print(f"Logical Feedback {i}:\n{logical_feedback}")

if __name__ == '__main__':
    test()
