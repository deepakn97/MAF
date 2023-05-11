import asyncio
import time
from typing import List
import pandas as pd
from tqdm import tqdm
from src.utils import Prompt, acall_gpt

from prompt_lib.backends import openai_api


class GSMInit(Prompt):
    def __init__(self, prompt_examples: str, engine: str, temperature: float, max_tokens: int = 300) -> None:
        super().__init__(
            question_prefix="# Q: ",
            answer_prefix="# solution using Python:\n",
            intra_example_sep="\n",
            inter_example_sep="\n\n",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, prompt_examples) -> str:
        with open(prompt_examples, "r") as f:
            self.prompt = f.read()
    
    def make_query(self, solution: str) -> str:
        solution = solution.strip()
        query = f"{self.prompt}{self.question_prefix}{solution}{self.intra_example_sep}{self.answer_prefix}"
        return query

    def __call__(self, solutions: List[str]) -> str:
        generation_queries = [self.make_query(solution) for solution in solutions]
        # print("Initial generation 0: ", generation_queries[0])
        # print("Initial generation 1: ", generation_queries[1])
        batch_size = 10
        async_responses = []
        for i in tqdm(range(0, len(generation_queries), batch_size), total=len(generation_queries)//batch_size):
            batch_responses = asyncio.run(
                acall_gpt(
                    generation_queries[i:i+batch_size],
                    self.engine,
                    self.temperature,
                    self.max_tokens,
                    stop_token="# Q:"
                )
            )
            async_responses.extend(batch_responses)
        
        solutions = []
        usage = 0
        finish_reason_stop = 0
        for response in async_responses:
            if "gpt" in self.engine:
                solutions.append(response['choices'][0]['message']['content'].strip())
            elif "text-davinci" in self.engine:
                solutions.append(response['choices'][0]['text'].strip())
                usage += response['usage']['total_tokens']
                finish_reason_stop += response['choices'][0]['finish_reason'] == "stop"
        print(f"Number of times the model finished because of stop token: {finish_reason_stop}/{len(async_responses)}")

        return usage, solutions


def test():
    task_init = GSMInit(
        prompt_examples="prompt/gsm_maf/init.txt",
        engine="text-davinci-003",
        temperature=0.0,
    )

    questions = ["Kelly is grocery shopping at a supermarket and is making sure she has enough in her budget for the items in her cart. Her 5 packs of bacon cost $10 in total and she has 6 packets of chicken which each cost twice as much as a pack of bacon. She also has 3 packs of strawberries, priced at $4 each, and 7 packs of apples, each priced at half the price of a pack of strawberries. If Kelly’s budget is $65 then how much money, in dollars, does she have left in her budget?", "Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?", "Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.", "Carly had 80 cards, 2/5 of the cards had the letter A on them, 1/2 of the remaining had the letter B, 5/8 of the rest had the letter C on them, and the others had the letter D. How many of the cards had the letter D on them?"]

    start = time.time()
    print(task_init(questions))
    end = time.time()
    print("Async version: ", end - start)

    start = time.time()
    for question in questions:
        print(task_init([question]))
    end = time.time()
    print("Sequential version: ", end - start)
    

if __name__ == "__main__":
    test()