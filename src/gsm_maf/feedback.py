import sys
import time
from typing import Dict

from src.utils import Prompt
import pandas as pd
from pathlib import Path
from prompt_lib.backends import openai_api


class MissingStepFeedback(Prompt):
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
        self.instruction = """# Check each semantically complete block of code for any missing steps and suggest the correct way to add them. Ignore all the other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def make_query(self, solution: str):
        solution = f"""{solution}{self.intra_example_sep}{self.instruction}"""
        return f"{self.prompt}{solution}"

    def __call__(self, solution: str):
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


class VariableNameFeedback(Prompt):
    pass


class CommonsenseFeedback(Prompt):
    pass


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


class GSMFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, temperature: float, max_tokens: int = 300) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n### END ###\n\n",
            engine=engine,
            temperature=temperature
        )
        self.max_tokens = max_tokens
        self.instruction = "# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."
        self.setup_prompt_from_examples_file(prompt_examples)


class LogicalFeedback(Prompt):
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
        self.instruction = """# Check each semantically complete block of the code to check for any logical reasoning errors. Logical reasoning errors may include errors in the mathematical calculations, errors in the order of the steps, or errors in the assumptions made. State the assumptions you made clearly. Ignore all the other types of errors."""
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        print(examples_path)
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def make_query(self, solution: str):
        solution = f"""{solution}{self.intra_example_sep}{self.instruction}"""
        return f"{self.prompt}{solution}"

    def __call__(self, solution: str):
        success = False
        while not success:
            try:
                generation_query = self.make_query(solution=solution)
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
        feedback = entire_output.strip()
        return feedback


class HallucinationFeedback(Prompt):
    pass


class CoherencyFeedback(Prompt):
    pass


def test():
    missing_step = MissingStepFeedback(
        prompt_examples="prompt/gsm_maf/missing_step.txt",
        engine="text-davinci-003",
        temperature=0.7
    )
    variable_naming = VariableNameFeedback(
        prompt_examples="prompt/gsm_maf/variable_naming.txt",
        engine="text-davinci-003",
        temperature=0.7,
    )
    logical = LogicalFeedback(
        prompt_examples="prompt/gsm_maf/logical.txt",
        engine="text-davinci-003",
        temperature=0.7,
    )

    wrong_soln = """def solution():
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
    return result"""
    vn_feedback_and_soln = variable_naming(wrong_soln)
    print(f"Variable Naming Feedback: {vn_feedback_and_soln['feedback']}")
    print(f"Variable Naming Solution: {vn_feedback_and_soln['solution']}")

    ms_feedback = missing_step(vn_feedback_and_soln['solution'])
    print(f"Missing Step Feedback: {ms_feedback}")

    logical_feedback = logical(vn_feedback_and_soln['solution'])
    print(f"Logical Feedback: {logical_feedback}")
    wrong_soln2 = """def solution():
    \"\"\"Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?\"\"\"
    flock_size = 3
    total_feed = 40 * flock_size
    final_meal = total_feed - 35
    return final_meal"""
    wrong_soln3 = """
    def solution():
    \"\"\"Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?\"\"\"
    morning_feed = 15
    afternoon_feed = 25
    total_feed = morning_feed + afternoon_feed
    feed_per_chicken_per_meal = total_feed / (20 * 3)
    final_meal_feed = feed_per_chicken_per_meal * 20 * 3
    return final_meal_feed"""
    wrong_soln4 = """
    def solution():
    \"\"\"Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?\"\"\"
    num_chickens = 20
    morning_feed = 15
    afternoon_feed = 25
    total_feed = num_chickens * 3 * 4
    final_feed = total_feed - morning_feed - afternoon_feed
    return final_feed"""
    wrong_soln5 = """
    def solution():
    \"\"\"Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?\"\"\"
    plates_cost = 0.5 * 12 * 6000  # cost of half a dozen plates
    cups_cost = (20 * 12 * x)  # total cost of 20 dozen cups
    
    # Using the given equation: cups_cost = plates_cost - 1200
    # Solving for x (cost of each cup):
    x = (plates_cost - 1200) / (20 * 12)
    return x
"""


if __name__ == '__main__':
    test()
