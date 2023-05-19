import os
import sys
import time
from typing import Dict, List
import torch

from tqdm import tqdm
from src.utils import ALPACA_MODEL_PATH, VICUNA_MODEL_PATH, Prompt, acall_gpt, call_gpt
import asyncio

from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.inference import load_model

class GSMIterate(Prompt):
    def __init__(self, engine: str, prompt_examples: str, temperature: float, max_tokens: int = 300) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n",
            engine=engine,
            temperature=temperature
        )
        self.max_tokens = max_tokens
        self.instruction = "# Given the feedback and the original code, let's rewrite the code to incorporate all of the feedback. Don't change anything unless it is mentioned in the feedback."
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()

    def __call__(self, solutions: List[str], feedbacks: Dict[str, List[str]], batch_size=10, concurrent=True) -> str:
            
        generation_queries = []
        for i in range(len(solutions)):
            solution = solutions[i]
            feedback = {}
            for ft, fb in feedbacks.items():
                feedback[ft] = fb[i]
            generation_queries.append(self.make_query(solution=solution, feedback=feedback))
        # print("Refined generation 0: ", generation_queries[0])
        # print("Refined generation 1: ", generation_queries[1])
        if not concurrent:
            batch_size = 1
        async_responses = []
        for i in tqdm(range(0, len(generation_queries), batch_size), total=len(generation_queries)//batch_size):
            if concurrent:
                batch_responses = asyncio.run(acall_gpt(
                    generation_queries[i:i+batch_size], 
                    self.engine, 
                    self.temperature, 
                    self.max_tokens,
                    stop_token="### END")
                )
            else:
                batch_responses = call_gpt(
                    generation_queries[i:i+batch_size],
                    self.engine,
                    self.temperature,
                    self.max_tokens,
                    stop_token="### END"
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
        query = f"{self.prompt}{self.intra_example_sep}{solution}{self.instruction}"
        return query
  
class OSIterate(Prompt):
    def __init__(self,
        prompt_examples: str = None,
        engine: str = "vicuna", 
        question_prefix: str = "",
        intra_example_sep: str = "\n\n",
        inter_example_sep: str = "\n\n",
        answer_prefix: str = "",
        model_device: str = "cuda",
        cuda_visible_devices: str = "0,1,2",
        max_gpu_memory: int = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        debug: bool = False,
        temperature: float = 0.0, 
        max_tokens: int = 300,
    ):
        super().__init__(
            question_prefix=question_prefix,
            answer_prefix=answer_prefix,
            intra_example_sep=intra_example_sep,
            inter_example_sep=inter_example_sep,
            engine=engine,
            temperature=temperature
        )
        self.max_tokens = max_tokens
        self.instruction = "# Given the feedback and the original code, let's rewrite the code to incorporate all of the feedback. Don't change anything unless it is mentioned in the feedback."
        self.setup_prompt_from_examples_file(prompt_examples)

        self.model_path = None
        if engine == "vicuna":
            self.model_path = VICUNA_MODEL_PATH
        elif engine == "alpaca":
            self.model_path = ALPACA_MODEL_PATH
        else:
            raise ValueError("Model name {engine} not supported. Choose between vicuna and alpaca")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        num_gpus = len(cuda_visible_devices.strip().split(","))

        self.model, self.tokenizer = load_model(
            self.model_path,
            model_device,
            num_gpus,
            max_gpu_memory,
            load_8bit,
            cpu_offloading,
            debug
        )

    def setup_prompt_from_examples_file(self, examples_path: str, **kwargs) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()
    
    def make_query(self, solution:str = None, feedback:Dict[str, str] = None, **kwargs) -> str:
        solution = f"""{solution}{self.intra_example_sep}"""
        for feedback_type, feedback_text in feedback.items():
            # if feedback_text != "":
            solution += f"""{feedback_type}:\n{feedback_text}{self.intra_example_sep}"""
        query = f"{self.prompt}{self.intra_example_sep}{solution}{self.instruction}"
        conv = get_conversation_template(self.model_path)
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()
        return query 
    
    def __call__(self, solutions: List[str], feedbacks: Dict[str, List[str]], batch_size=10, concurrent=True) -> str:
        generation_queries = []
        for i in range(len(solutions)):
            solution = solutions[i]
            feedback = {}
            for ft, fb in feedbacks.items():
                feedback[ft] = fb[i]
            generation_queries.append(self.make_query(solution=solution, feedback=feedback))
        entire_outputs = []

        for i in tqdm(range(len(generation_queries)), total=len(generation_queries)):
            print(f"GPU Memory 0: {torch.cuda.memory_allocated(0)/1e9} GB")
            print(f"GPU Memory 1: {torch.cuda.memory_allocated(1)/1e9} GB")
            print(f"GPU Memory 2: {torch.cuda.memory_allocated(2)/1e9} GB")

            input_ids = self.tokenizer([generation_queries[i]]).input_ids
            input_ids = torch.as_tensor(input_ids).to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens
                )

            if self.model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids):]
            
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
            entire_outputs.append(output)
            del input_ids
            del output_ids
        torch.cuda.empty_cache()

        solutions = []
        for entire_output in entire_outputs:
            if "### END ###" in entire_output:
                entire_output = entire_output.split("### END ###")[0].strip()
            solution = ""
            if "def solution():" in entire_output:
                solution = entire_output.split("def solution():")[1]
                solution = "def solution():" + solution.rstrip()
            solutions.append(solution)
        return solutions

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
    "Missing Step Feedback": [
    """# Let us go through the code step-by-step
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

    os_task_iterate = OSIterate(
        engine='vicuna',
        prompt_examples='prompt/gsm_maf/iterate.txt',
        temperature=0.0
    )
    start = time.time()
    print(os_task_iterate(solutions, feedbacks))


if __name__ == "__main__":
    test()