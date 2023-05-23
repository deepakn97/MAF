import asyncio
import math
import os
import time
from typing import List
import pandas as pd
import json
import torch
import baseline_utils
from tqdm import tqdm
from src.utils import Prompt, acall_gpt, call_gpt, VICUNA_MODEL_PATH, ALPACA_MODEL_PATH, get_gpu_memory
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.inference import load_model
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
        return query 
    
    def __call__(self, solutions: List[str], batch_size=10, concurrent=True) -> str:
        generation_queries = [self.make_query(solution) for solution in solutions]
        async_responses = []
        print(generation_queries)
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
            async_responses.append(output)
            del input_ids
            del output_ids
        torch.cuda.empty_cache()
        return async_responses

def test():



    os_task_init = OSInit(
        prompt_examples="../../prompt/os_maf/init.txt",
        engine="vicuna",
        temperature=0.01
    )

    start = time.time()
    response = os_task_init(questions)
    with open('output1.txt', 'w') as f:
        f.write(json.dumps(response))
    end = time.time()
    print("Vicuna version: ", end - start)
    

if __name__ == "__main__":
    test()