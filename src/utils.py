from abc import ABCMeta, abstractmethod
import math
import os
import time
import traceback
import openai
import asyncio
import backoff
from typing import Callable, Dict, List, Union
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.inference import load_model

import numpy as np
import torch
from tqdm import tqdm

VICUNA_MODEL_PATH = "/home/ubuntu/vicuna_weights/13B"
ALPACA_MODEL_PATH = "/data4/dnathani/alpaca-7b"

class Prompt:
    def __init__(
        self,
        question_prefix: str,
        answer_prefix: str,
        intra_example_sep: str,
        inter_example_sep: str,
        engine: str = None,
        temperature: float = None,
    ) -> None:
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.intra_example_sep = intra_example_sep
        self.inter_example_sep = inter_example_sep
        self.engine = engine
        self.temperature = temperature

    def make_query(self, prompt: str, question: str) -> str:
        return (
            f"{prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        )


class Feedback(metaclass=ABCMeta):
    def __init__(
        self,
        name: str = "Feedback",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name

    @abstractmethod
    def __call__(self, solutions: List[str], **kwargs) -> Union[str, Dict[str, str]]:
        """Call the feedback module on the solution and return a feedback string or a dictionary of outputs."""

class FeedbackFactory:
    """ The factory class for feedback generation. """
    registry = {}

    @classmethod
    def create_feedback(cls, name: str, **kwargs) -> 'Feedback':
        feedback_class = cls.registry[name]
        feedback = feedback_class(**kwargs)
        return feedback

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Feedback) -> Callable:
            if name in cls.registry:
                print(f"Feedback {name} already exists. Will replace it.")
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
class LLMFeedback(Feedback):
    def __init__(
        self,
        engine: str = "text-davinci-002",
        question_prefix: str = "# Q: ",
        answer_prefix: str = "# A:",
        intra_example_sep: str = "\n\n",
        inter_example_sep: str = "\n\n",
        temperature: float = 0.0,
        max_tokens: int = 300,
        eager_refine: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.engine = engine
        self.temperature = temperature
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.intra_example_sep = intra_example_sep
        self.inter_example_sep = inter_example_sep
        self.max_tokens = max_tokens
        self.eager_refine = eager_refine
        self.instruction = "# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."

    def setup_prompt_from_examples_file(self, examples_path: str, **kwargs) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()
    
    def make_query(self, solution: str, **kwargs) -> str:
        solution = f"""{solution}{self.intra_example_sep}{self.instruction}"""
        return f"{self.prompt}{solution}"
    
    def process_outputs(self, outputs: List[str], **kwargs) -> List[str]:
        fb_and_solns = []
        for entire_output in outputs:
            if "### END" in entire_output:
                entire_output = entire_output.split("### END")[0]
            fb_and_maybe_soln = entire_output.strip()
            if self.eager_refine:
                if self.answer_prefix in fb_and_maybe_soln:
                    feedback = fb_and_maybe_soln.split(self.answer_prefix)[0].strip()
                    solution = fb_and_maybe_soln.split(self.answer_prefix)[1].rstrip()
                    solution = f"{self.answer_prefix}{solution}"
                else:
                    feedback = fb_and_maybe_soln
                    solution = ""
            else:
                feedback = fb_and_maybe_soln
                solution = ""
            
            fb_and_solns.append({"feedback": feedback, "solution": solution})

        return fb_and_solns

    def __call__(self, solutions: List[str], batch_size=10, concurrent=True):
        generation_queries = [self.make_query(solution) for solution in solutions]
        if not concurrent:
            batch_size = 1

        async_responses = []
        # print("Feedback solutions length: ", len(solutions))
        # print(len(generation_queries))
        # print(batch_size)
        for i in tqdm(range(0, len(generation_queries), batch_size), total=len(generation_queries)//batch_size):
            if concurrent:
                batch_responses = asyncio.run(
                    acall_gpt(
                        generation_queries[i:i+batch_size], 
                        self.engine, 
                        self.temperature, 
                        self.max_tokens,
                        stop_token="### END"
                    )
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
                usage += response['usage']['total_tokens']
                finish_reason_stop += response['choices'][0]['finish_reason'] == "stop"
            elif "text-davinci" in self.engine:
                entire_outputs.append(response['choices'][0]['text'].strip())
                usage += response['usage']['total_tokens']
                finish_reason_stop += response['choices'][0]['finish_reason'] == "stop"
        print(f"Number of times the model finished because of stop token: {finish_reason_stop}/{len(async_responses)}")
        
        fb_and_solns = self.process_outputs(entire_outputs)
        # print(entire_outputs)

        return usage, fb_and_solns

class OSModel():
    """This class is meant to implement common functions to call all Open-source based LLMs. These common functions include make_query, __call__, setup_prompt_from_examples_file, load_model"""
    pass

class LLMModel():
    """This class is meant to implement common functions to call all API based LLMs. These common functions include make_query, __call__, and setup_prompt_from_examples_file"""
    pass

class OSFeedback(Feedback):
    def __init__(
        self,
        engine: str = "vicuna", 
        question_prefix: str = "# Q: ",
        intra_example_sep: str = "\n\n",
        inter_example_sep: str = "### END ###",
        answer_prefix: str = "def solution():",
        eager_refine: bool = False,
        model_device: str = "cuda",
        cuda_visible_devices: str = "0,1,2",
        max_gpu_memory: int = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        debug: bool = False,
        temperature: float = 0.0, 
        max_tokens: int = 300,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.engine = engine
        self.temperature = temperature
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.intra_example_sep = intra_example_sep
        self.inter_example_sep = inter_example_sep
        self.max_tokens = max_tokens
        self.eager_refine = eager_refine
        self.instruction = "# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."

        self.model_path = None
        if engine == "vicuna":
            self.model_path = VICUNA_MODEL_PATH
        elif engine == "alpaca":
            self.model_path = ALPACA_MODEL_PATH
        else:
            raise ValueError("Model name {engine} not supported. Choose between vicuna and alpaca")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        num_gpus = len(cuda_visible_devices.strip().split(","))

        # cap gpu memory usage to 60% for the model
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
    
    def make_query(self,
        solution:str = None, 
    ) -> str:
        query = f"""{self.prompt}{solution}{self.intra_example_sep}{self.instruction}"""
        # conv = get_conversation_template(self.model_path)
        # conv.append_message(conv.roles[0], query)
        # conv.append_message(conv.roles[1], None)
        # query = conv.get_prompt()
        return query 

    def process_outputs(self, outputs: List[str]) -> List[str]:
        """Implementation for processing outputs from the model. This function is meant to be overriden by subclasses for different datasets."""
        fb_and_solns = []
        for entire_output in outputs:
            if "### END" in entire_output:
                entire_output = entire_output.split("### END")[0]
            fb_and_maybe_soln = entire_output.strip()
            if self.eager_refine:
                if self.answer_prefix in fb_and_maybe_soln:
                    feedback = fb_and_maybe_soln.split(self.answer_prefix)[0].strip()
                    solution = fb_and_maybe_soln.split(self.answer_prefix)[1].rstrip()
                    solution = f"{self.answer_prefix}{solution}"
                else:
                    feedback = fb_and_maybe_soln
                    solution = ""
            else:
                feedback = fb_and_maybe_soln
                solution = ""
            
            fb_and_solns.append({"feedback": feedback, "solution": solution})

        return fb_and_solns

    def __call__(self, solutions: List[str], batch_size=10, concurrent=True) -> List[str]:
        generation_queries = [self.make_query(solution) for solution in solutions]
        entire_outputs = []

        for i in tqdm(range(len(generation_queries)), total=len(generation_queries)):
            # print(generation_queries[i])

            input_ids = self.tokenizer([generation_queries[i]]).input_ids
            input_ids = torch.as_tensor(input_ids).to(self.model.device)
            with torch.no_grad():
                if self.temperature == 0.0:
                    output_ids = self.model.generate(
                        input_ids,
                        do_sample=False,
                        max_new_tokens=self.max_tokens
                    )
                else:
                    output_ids = self.model.generate(
                        input_ids,
                        do_sample=True,
                        temperature=self.temperature,
                        max_new_tokens=self.max_tokens
                    )

            if self.model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]):]
            print(f"Output tokens: {len(output_ids)}")
            
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
            entire_outputs.append(output)
            del input_ids
            del output_ids
        torch.cuda.empty_cache()

        fb_and_solns = self.process_outputs(entire_outputs)

        return fb_and_solns

class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.items():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(str(msg) + '\n')
        self.log_file.flush()
        print(msg)

def parse_feedback(feedback):
    feedback = feedback.split("\n\n")
    feedback = [f.rstrip() for f in feedback if '# looks good' not in f.strip().lower()]
    return "\n\n".join(feedback)

def backoff_handler(details):
    print(f"Try: {details['tries']}, waiting {details['wait']} because: {details['exception']}")

@backoff.on_exception(backoff.expo, Exception, max_tries=20, raise_on_giveup=False, on_backoff=backoff_handler)
async def acall_gpt(
    queries: List[str], 
    engine: str = "text-davinci-002", 
    temperature: float = 0.0, 
    max_tokens: int = 300, 
    stop_token:str = "### END"
) -> List[Dict]:
    if "gpt" in engine:
        async_responses = [
            openai.ChatCompletion.acreate(
                model=engine,
                messages=[{"role": "user", "content": query}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_token
            )
            for query in queries
        ]
    elif "text-davinci" in engine:
        # print("Temperature: ", temperature)
        # print("Max tokens: ", max_tokens)
        # print("engine: ", engine)
        async_responses = [
            openai.Completion.acreate(
                model=engine,
                prompt=query,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=[stop_token]
            )
            for query in queries
        ]

    return await asyncio.gather(*async_responses)

@backoff.on_exception(backoff.expo, Exception, max_tries=20, raise_on_giveup=False, on_backoff=backoff_handler)
def call_gpt(
    queries: List[str],
    engine: str = "text-davinci-002",
    temperature: float = 0.0,
    max_tokens: int = 300,
    stop_token: str = "### END"
) -> List[Dict]:
    responses = []
    if "gpt" in engine:
        queries = [{"role": "user", "content": query} for query in queries]
        for query in queries:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=query,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_token
            )
            responses.append(response)
    elif "text-davinci" in engine:
        for query in queries:
            response = openai.Completion.create(
                model=engine,
                prompt=query,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=[stop_token]
            )
            responses.append(response)

    return responses

def retry_parse_fail_prone_cmd(
    func,
    max_retries: int = 100,
    exceptions=(
        ValueError,
        KeyError,
        IndexError,
    ),
):
    def wrapper(*args, **kwargs):
        retries = max_retries
        while retries:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                stack_trace = traceback.format_exc()

                retries -= 1
                print(
                    f"An error occurred: {e}. {stack_trace}. Left retries: {retries}.")
        return None

    return wrapper

def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return max(gpu_memory)
