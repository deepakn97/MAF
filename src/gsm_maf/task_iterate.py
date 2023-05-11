import sys
import time
from typing import Dict, List

from tqdm import tqdm
from src.utils import Prompt, acall_gpt
import asyncio

from prompt_lib.backends import openai_api

class GSMIterate(Prompt):
    def __init__(self, engine: str, prompt_examples: str, temperature: float, max_tokens: int = 300) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n",
        )
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.instruction = "# Given the feedback and the original code, let's rewrite the code to incorporate all of the feedback. Don't change anything unless it is mentioned in the feedback."
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def __call__(self, solutions: List[str], feedbacks: Dict[str, List[str]]) -> str:
            
        generation_queries = []
        for i in range(len(solutions)):
            solution = solutions[i]
            feedback = {}
            for ft, fb in feedbacks.items():
                feedback[ft] = fb[i]
            generation_queries.append(self.make_query(solution=solution, feedback=feedback))
        # print("Refined generation 0: ", generation_queries[0])
        # print("Refined generation 1: ", generation_queries[1])
        batch_size = 10
        async_responses = []
        for i in tqdm(range(0, len(generation_queries), batch_size), total=len(generation_queries)//batch_size):
            batch_responses = asyncio.run(acall_gpt(
                generation_queries[i:i+batch_size], 
                self.engine, 
                self.temperature, 
                self.max_tokens,
                stop_token="### END ###")
            )
            async_responses.extend(batch_responses)
        entire_outputs = []
        usage = 0
        finish_reason_stop = 0
        for response in async_responses:
            if "gpt" in self.engine:
                entire_outputs.append(response['choices'][0]['message']['content'].strip())
            elif "text-davinci" in self.engine:
                entire_outputs.append(response['choices'][0]['text'].strip())
                usage += response['usage']['total_tokens']
                finish_reason_stop += response['choices'][0]['finish_reason'] == "stop"
        print(f"Number of times the model finished because of stop token: {finish_reason_stop}/{len(async_responses)}")

        # print(f"Iterate Output: {entire_output}")
        solutions = []
        for entire_output in entire_outputs:
            if "### END ###" in entire_output:
                entire_output = entire_output.split("### END ###")[0].strip()
            solution = ""
            if "def solution():" in entire_output:
                solution = entire_output.split("def solution():")[1]
                solution = "def solution():" + solution.rstrip()
            solutions.append(solution)
        return usage, solutions

    def make_query(self, solution: str, feedback: Dict[str, str]) -> str:
        solution = f"""{solution}{self.intra_example_sep}"""
        for feedback_type, feedback_text in feedback.items():
            # if feedback_text != "":
            solution += f"""{feedback_type}:\n{feedback_text}{self.intra_example_sep}"""
        query = f"{self.prompt}{self.intra_example_sep}{solution}"
        return query
  
def test():
    task_iterate = GSMIterate(
    engine="text-davinci-003",
    prompt_examples="prompt/gsm_maf/iterate.txt",
    temperature=0.7
    )

    wrong_solns = ["""def solution():
    \"\"\"Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?\"\"\"
    chips_per_square_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
    chips_needed = height * chips_per_square_inch
    chips_available = bags * chips_per_bag
    chips_left = chips_available - chips_needed
    length = chips_left / chips_per_square_inch
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
    feedbacks = {
    "Missing Step Feedback": ["""# Let us go through the code step-by-step
    chips_per_square_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
    # looks good

    # Let's check other parts
    chips_needed = height * chips_per_square_inch
    chips_available = bags * chips_per_bag
    # wrong! we need to caclulate the area of the mosaic which can be made by available chips. This can be calculated by dividing chips available with chips per square inch. Let's add it!

    # Let's check other parts
    chips_left = chips_available - chips_needed
    length = chips_left / chips_per_square_inch
    result = length
    return result
    # looks good""",
    """# Let us go through the code step-by-step
    budget = 65
# looks good

# Let's check other parts
    bacon_packs = 5
    bacon_total_cost = 10
# looks good

# Let's check other parts
    chicken_packs = 6
    chicken_cost = 2 * bacon_cost
# wrong! bacon_cost is missing. Let's add it.
# wrong! we need the total cost of chicken to calculate remaining budget. Let's add it.
    chicken_packs = 6
    chicken_cost = 2 * bacon_cost
    
# Let's check other parts
    strawberry_packs
    strawberry_cost = 4
# wrong! we need the total cost of strawberries to calculate remaining budget. Let's add it.

# Let's check other parts
    apple_packs = 7
    apple_cost = strawberry_cost / 2
# wrong! we need the total cost of apples to calculate remaining budget. Let's add it.

# Let's check other parts
    total_cost = bacon_cost + chicken_cost + strawberry_cost + apple_cost
    money_left = budget - total_cost
    result = money_left
    return result
# looks good"""
    ],
    "Logical Reasoning Feedback": ["""# Let us go through the code step-by-step
    chips_per_square_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
    # looks good

    # Let's check other parts
    chips_needed = height * chips_per_square_inch
    chips_available = bags * chips_per_bag
    # wrong! chips needed doesn't make sense here. remove it

    # Let's check other parts
    chips_left = chips_available - chips_needed
    length = chips_left / chips_per_square_inch
    result = length
    return result
    # wrong! chips_left doesn't make sense here. remove it
    # wrong! we want to divide the *area of the mosaic* by the height of the mosaic to get the length of the mosaic. Let's fix it!""",
    """# Let us go through the code step-by-step
    budget = 65
# looks good

# Let's check other parts
    bacon_packs = 5
    bacon_total_cost = 10
# looks good

# Let's check other parts
    chicken_packs = 6
    chicken_cost = 2 * bacon_total_cost
# wrong! according to the context, the cost of each packet of chicken is twice the cost of 1 packet of bacon. We should use bacon_cost in place of bacon_total_cost to calculate the chicken pack cost correctly. Let's fix it.
    chicken_packs = 6
    chicken_cost = 2 * bacon_cost

# Let's check other parts
    strawberry_packs
    strawberry_cost = 4
# looks good

# Let's check other parts
    apple_packs = 7
    apple_cost = strawberry_cost / 2
# looks good

# Let's check other parts
    total_cost = bacon_cost + chicken_cost + strawberry_cost + apple_cost
    money_left = budget - total_cost
    result = money_left
    return result
# wrong! we want to calculate the total cost of buying all the items so we should use the total cost of each item instead of cost of 1 pack of each item. Let's fix it."""
    ],
}
    start = time.time()
    print(task_iterate(wrong_solns, feedbacks))
    end = time.time()
    print("Async version: ", end - start)

    solutions = []
    fbs = []
    for i in range(len(wrong_solns)):
        solution = wrong_solns[i]
        feedback = {}
        for ft, fb in feedbacks.items():
            feedback[ft] = [fb[i]]
        solutions.append(solution)
        fbs.append(feedback)

    start = time.time()
    for soln, fb in zip(wrong_solns, fbs):
        print(task_iterate([soln], fb))
    end = time.time()
    print("Sequential version: ", end - start)


if __name__ == "__main__":
    test()