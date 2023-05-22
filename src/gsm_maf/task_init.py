import asyncio
import math
import os
import time
from typing import List
import pandas as pd
import torch
from tqdm import tqdm
from src.utils import Prompt, acall_gpt, call_gpt, VICUNA_MODEL_PATH, ALPACA_MODEL_PATH, get_gpu_memory
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.inference import load_model

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

    def __call__(self, solutions: List[str], batch_size=10, concurrent=True) -> str:
        generation_queries = [self.make_query(solution) for solution in solutions]
        # print("initial generation query 0:\n", generation_queries[0])
        # print("initial generation query 1:\n", generation_queries[1])
        if not concurrent:
            batch_size = 1

        async_responses = []
        for i in tqdm(range(0, len(generation_queries), batch_size), total=len(generation_queries)//batch_size):
            if concurrent:
                batch_responses = asyncio.run(
                    acall_gpt(
                        generation_queries[i:i+batch_size],
                        self.engine,
                        self.temperature,
                        self.max_tokens,
                        stop_token=self.inter_example_sep
                    )
                )
            else:
                # print(f"{i}th generation_query:\n{generation_queries[i:i+batch_size]}")
                # print(self.engine)
                # print(self.temperature)
                # print(self.max_tokens)
                batch_responses = call_gpt(
                    generation_queries[i:i+batch_size],
                    self.engine,
                    self.temperature,
                    self.max_tokens,
                    stop_token=self.inter_example_sep
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
        # print("Init Solutions length: ", len(solutions))

        return usage, solutions

class OSInit(Prompt):
    def __init__(self,
        prompt_examples: str = None,
        engine: str = "vicuna", 
        question_prefix: str = "# Q: ",
        intra_example_sep: str = "\n\n",
        inter_example_sep: str = "\n\n",
        answer_prefix: str = "# A:",
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
        self.setup_prompt_from_examples_file(prompt_examples)

        self.model_path = None
        if engine == "vicuna":
            self.model_path = VICUNA_MODEL_PATH
        elif engine == "alpaca":
            self.model_path = ALPACA_MODEL_PATH
        else:
            raise ValueError("Model name {engine} not supported. Choose between vicuna and alpaca")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(os.environ["CUDA_VISIBLE_DEVICES"])
        num_gpus = len(cuda_visible_devices.strip().split(","))

        if max_gpu_memory is None:
            max_gpu_memory = str(int(math.ceil(get_gpu_memory(num_gpus) * 0.99))) + "GiB"

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
    
    def make_query(self, solution:str = None, **kwargs) -> str:
        query = f"""{self.prompt}{solution}{self.inter_example_sep}"""
        conv = get_conversation_template(self.model_path)
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()
        print(query)
        return query 
    
    def __call__(self, solutions: List[str], batch_size=10, concurrent=True) -> str:
        generation_queries = [self.make_query(solution) for solution in solutions]
        async_responses = []

        for i in tqdm(range(len(generation_queries)), total=len(generation_queries)):
            # print(f"GPU Memory 0: {torch.cuda.memory_allocated(0)/1e9} GB")
            # print(f"GPU Memory 1: {torch.cuda.memory_allocated(1)/1e9} GB")
            # print(f"GPU Memory 2: {torch.cuda.memory_allocated(2)/1e9} GB")

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
            async_responses.append(output)
            del input_ids
            del output_ids
        torch.cuda.empty_cache()
        return async_responses

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

    os_task_init = OSInit(
        prompt_examples="prompt/gsm_maf/init.txt",
        engine="vicuna",
        temperature=1.0
    )

    start = time.time()
    print(os_task_init(questions[0:1]))
    end = time.time()
    print("Vicuna version: ", end - start)
    

if __name__ == "__main__":
    test()