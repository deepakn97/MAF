import time
import pandas as pd
from src.utils import Prompt

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

    def __call__(self, solution: str) -> str:
        generation_query = self.make_query(solution)
        success = False
        while not success:
            try:
                output = openai_api.OpenaiAPIWrapper.call(
                    prompt=generation_query,
                    engine=self.engine,
                    max_tokens=self.max_tokens,
                    stop_token=self.inter_example_sep,
                    temperature=self.temperature,
                )
                success = True
            except Exception as e:
                success = False
                print(e)
                time.sleep(60)

        solution_code = openai_api.OpenaiAPIWrapper.get_first_response(output)

        return solution_code.strip()


def test():
    task_init = GSMInit(
        prompt_examples="prompt/gsm_maf/init.txt",
        engine="text-davinci-003",
        temperature=0.0,
    )

    question = "Kelly is grocery shopping at a supermarket and is making sure she has enough in her budget for the items in her cart. Her 5 packs of bacon cost $10 in total and she has 6 packets of chicken which each cost twice as much as a pack of bacon. She also has 3 packs of strawberries, priced at $4 each, and 7 packs of apples, each priced at half the price of a pack of strawberries. If Kelly’s budget is $65 then how much money, in dollars, does she have left in her budget?"
    print(task_init(question))
    

if __name__ == "__main__":
    test()