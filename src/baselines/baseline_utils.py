import os
import json
import nltk
import numpy as np
import re
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from io import StringIO
from contextlib import redirect_stdout
from src.utils import Prompt, acall_gpt, call_gpt, VICUNA_MODEL_PATH, ALPACA_MODEL_PATH, get_gpu_memory
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.inference import load_model
import math
import torch
from langchain.llms import OpenAI
import time
from typing import List
from tqdm import tqdm
from src.gsm_iter.gsm_selfref_eval import check_corr
import asyncio
from time import sleep
class OSModel(Prompt):
    def __init__(self,
        prompt_examples: str = None,
        engine: str = "vicuna", 
        question_prefix: str = "# Q: ",
        intra_example_sep: str = "\n\n",
        inter_example_sep: str = "\n\n",
        answer_prefix: str = "# A:",
        model_device: str = "cuda",
        cuda_visible_devices: str = "0, 1",
        max_gpu_memory: int = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        debug: bool = False,
        temperature: float = 0.0, 
        max_tokens: int = 512,
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
        self.prompt = ''

        self.model_path = None
        if engine == "vicuna":
            self.model_path = VICUNA_MODEL_PATH
        elif engine == "alpaca":
            self.model_path = ALPACA_MODEL_PATH
        else:
            raise ValueError(f'Model name {engine} not supported. Choose between vicuna and alpaca')
        
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(os.environ["CUDA_VISIBLE_DEVICES"])
        num_gpus = len(cuda_visible_devices.strip().split(","))

        if max_gpu_memory is None:
            max_gpu_memory = str(int(math.ceil(get_gpu_memory(num_gpus) * 0.8))) + "GiB"

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
        return query 
    
    


    async def __call__(self, solutions: List[str], batch_size=10, concurrent=True) -> str:
        generation_queries = [self.make_query(solution) for solution in solutions]
        async_responses = []
        eos_token = "USER:"
        eos_token_id = self.tokenizer.encode(eos_token, add_special_tokens=False)[0]
        for i in tqdm(range(len(generation_queries)), total=len(generation_queries)):
            # print(f"GPU Memory 0: {torch.cuda.memory_allocated(0)/1e9} GB")
            # print(f"GPU Memory 1: {torch.cuda.memory_allocated(1)/1e9} GB")
            # print(f"GPU Memory 2: {torch.cuda.memory_allocated(2)/1e9} GB")


            # time how long tokenizing takes
            input_ids = self.tokenizer([generation_queries[i]]).input_ids
            print('len input_ids', len(input_ids))
            print('len input_ids[0]', len(input_ids[0]))
            input_ids = torch.as_tensor(input_ids).to(self.model.device)
            
            # time how long generation takes
            # add eos token 'USER: ' to stop Vicuna from generating user responses to continue conversation
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens,
                    early_stopping=True,
                    eos_token_id=eos_token_id,

                )

            if self.model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]):]
            
            # time how long decoding takes
            output = self.tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
            async_responses.append(output)
            del input_ids
            del output_ids
        torch.cuda.empty_cache()
        return async_responses
    def clear_cache(self):
        with torch.no_grad():
            torch.cuda.empty_cache()
        

class OpenAIWrapper:
    def __init__(self, **kwargs):
        self.llm = OpenAI(**kwargs)
    async def __call__(self, prompts):
        outputs = [self.async_generate_answer(prompt) for prompt in prompts]
        return await asyncio.gather(*outputs)
    async def async_generate_answer(self, prompt):
        success = False
        while not success:
            try:
                output = await self.llm.agenerate([prompt])
                success = True
            except Exception as e:
                print(e)
                print(f'API server overloaded. Waiting for 30 seconds...')
                sleep(30)
                continue
        return output.generations[0][0].text

async def async_generate_answers(llm, prompt_template, problems):
  '''Generate the answer for the given problem.'''
  outputs = await llm([prompt_template.format(context = prob) for prob in problems])
  return outputs

async def gsm_run(prompt_template, llm, data):
    problems = [d['input'] for d in data]
    step = 5
    for i in range(0, len(problems), step):
        outputs = await async_generate_answers(llm, prompt_template, problems[i:min(i + step, len(problems))])
        for j in range(step):
            if (i + j) < len(problems):
                data[i + j]['output'] = outputs[j]
        print (f"Completed {i + step} problems")
    return data

async def gsm_baseline(llm, task, prompt_technique, model_name, data_dir, save_dir):
    save_dir = os.path.join(save_dir, f'{model_name}/{prompt_technique}')
    prompt_template = create_prompt_template(task, model_name, prompt_technique)
    for i in range(3):
        for variant in ['original', 'irc']:
            data = load_gsm_data(os.path.join(data_dir, f'gsmic_mixed_{i}_{variant}.jsonl'))

            if (not os.path.exists(save_dir)):
                os.makedirs(save_dir)
            if (os.path.exists(os.path.join(save_dir, f'gsmic_mixed_{i}_{variant}_output_{model_name}_{prompt_technique}.json')) 
                or os.path.exists(os.path.join(save_dir, f'hand_gsmic_mixed_{i}_{variant}_output_{model_name}_{prompt_technique}.json'))):
                continue
            problems = await gsm_run(prompt_template, llm, data)
            output_file = os.path.join(save_dir, f'gsmic_mixed_{i}_{variant}_output_{model_name}_{prompt_technique}.json')
            print(output_file)
            with open(output_file, 'w') as f:
                  f.write(json.dumps(problems) + '\n')


def grade(filepath, rewrite = False):
        with open(filepath, 'r') as f:
            problems = json.load(f)
        for p in problems:
            if (p['input'] == 'undefined'):
                p['correct'] = False
            elif (check_corr(float(p['input']), float(p['target']))):
                p['correct'] = True
            else:
                p['correct'] = False
        if (rewrite):
            filepath = os.path.join('/'.join(filepath.split('/')[:-1]), 'hand_' + filepath.split('/')[-1])
        with open(filepath, 'w') as f:
            f.write(json.dumps(problems) + '\n')
        print(calc_accuracy(problems))

def parse_answers(filepath, task):
    with open(filepath, 'r') as f:
        problems = json.load(f)
    for p in problems:
        p['answer'] = parse_answer(p['output'], task)
    with open(filepath, 'w') as f:
        f.write(json.dumps(problems) + '\n')

def load_jsonl(filepath):
    '''Loads jsonl file into list of dict objects'''

def load_gsm_data(filepath):
    '''Loads jsonl file of all questions and associated answers into list of dict objects with following keys:
    'input', 'target', 'n_steps'
    '''
    return load_jsonl(filepath)



def create_prompt_template(task, model_name, prompt_technique):
    with open(os.path.join(f'prompt/{task}/{model_name}', f'{prompt_technique}.txt'), 'r') as f:
        template = f.read()
    prompt = PromptTemplate(
        input_variables=['context'], template=template)
    return prompt

def calc_accuracy(problems):
    return sum([p['correct'] for p in problems]) / len(problems)

def check_corr(input: str, target: str, tol: float = 0.001):
    try:
        return (abs(float(input) - float(target)) < tol)
    except:
        return False

def manual_parse(filename):
    with open(filename, 'r') as f:
        problems = json.load(f)
    for problem in problems:
        # check if target is anywhere in output
        if problem['target'] in problem['output']:
            problem['final_answer'] = 'undefined'
        else:
            problem['final_answer'] = 'wrong' 
    with open(filename, 'w') as f:
        f.write(json.dumps(problems, indent=4))


def parse_program(answer):
    # Takes code that ends in print statement. Returns output of print statement
    answer = answer.split('\n')
    # if there is a line with def solution(), only take the lines inside the function
    for i, line in enumerate(answer):
        if (line.lower().find('def solution') != -1):
            answer = answer[i + 1:]
            break
    # Remove spaces from beginning of each line so that indentation is not an issue
    answer = [line.lstrip() for line in answer]
    # Remove lines after print statement
    for i, line in enumerate(answer):
        if (line.lower().find('print') != -1):
            answer = answer[:i + 1]
            break
    return '\n'.join(answer)
def parse_program_answer(answer):
    # Execute code and capture print output
    answer = parse_program(answer)
    f = StringIO()
    try:
        with redirect_stdout(f):
            exec(answer)
        answer = f.getvalue()
        answer = answer.split('\n')[0]
        answer = ''.join(c for c in answer if c.isdigit() or c == '.')
        return answer
    except:
        return 'undefined'
    

def parse_answer(answer, task):
    if task == 'pot_gsm':
        return parse_program_answer(answer)
    else:
        answer_key = 'final_answer: '
        if (answer.find(answer_key) == -1):
            return "undefined"
        answer = answer[answer.find(answer_key) + len(answer_key):]
        answer = answer.split('\n')[0]
        answer = ''.join(c for c in answer if c.isdigit() or c ==
                         '.')  # note that commas are removed
        try:
            fl_answer = float(answer)
            int_answer = int(fl_answer)
            if (fl_answer == int_answer):
                return str(int_answer)
            else:
                return str(fl_answer)
        except:
            return "undefined"


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
