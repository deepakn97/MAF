from abc import ABCMeta, abstractmethod
import os
import time
import traceback
import openai
import asyncio
import backoff
from typing import Callable, Dict, List, Union

import numpy as np
from prompt_lib.backends import openai_api

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
    def __call__(self, solution: str, **kwargs) -> Union[str, Dict[str, str]]:
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
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=10)
    async def async_requests(self, queries: List[str]):
        if "gpt" in self.engine:
            async_responses = [
                openai.ChatCompletion.acreate(
                    model=self.engine,
                    messages=query,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop="### END"
                )
                for query in queries
            ]
        elif "text-davinci" in self.engine:
            async_responses = [
                openai.Completion.acreate(
                    model=self.engine,
                )
                for query in queries
            ]

        return await asyncio.gather(*async_responses)

    
    def __call__(self, solutions: List[str], batch: bool = False):
        generation_query = self.make_query(solution=solution)
        success = False

        output = openai_api.OpenaiAPIWrapper.call(
            prompt = generation_query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="### END",
            temperature=self.temperature,
        ) 
        success = True

        entire_output = openai_api.OpenaiAPIWrapper.get_first_response(output)
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
        
            return {"feedback": feedback, "solution": solution}
        return fb_and_maybe_soln

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
    feedback = [f for f in feedback if f.split(
        '\n')[-1].lower() != '# looks good']
    return "\n\n".join(feedback)