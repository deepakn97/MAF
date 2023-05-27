import os
import json
import nltk
import numpy as np
import re
from langchain.prompts import FewShotPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage
from io import StringIO
from contextlib import redirect_stdout
from src.utils import Prompt, acall_gpt, call_gpt, VICUNA_MODEL_PATH, ALPACA_MODEL_PATH, OSModel, LLMModel
from langchain.chat_models import ChatOpenAI
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.inference import load_model
import math
import torch
from  src.entailment.utils import get_entailment_proof
from langchain.llms import OpenAI
import time
from typing import List
from tqdm import tqdm
from src.gsm_iter.gsm_selfref_eval import check_corr
import asyncio
from time import sleep

OS_MODELS = ["vicuna", "alpaca"]
OPENAI_MODELS = ["gpt-3.5-turbo", "text-davinci-003"]
TASKS = ['gsm_baseline', 'entailment_baseline']
PROMPTS = ['0cot_gsm', '1cot_gsm', '4cot_gsm', 'pot_gsm', 'ltm_gsm', '3shot_entailment']

class BaselineWrapper:
    def __init__(
        self,
        engine: str = 'text-davinci-003',
        task: str = 'gsm_baseline',
        prompt: str = 'pot_gsm',
        data_dir: str = None,
        save_dir: str = 'src/baselines/models',
        llm = None,
        **kwargs
    ):
        if (task not in TASKS):
            raise ValueError(f"Invalid task {task}")
        if (prompt not in PROMPTS):
            raise ValueError(f"Invalid prompt {prompt}")
        if llm is not None:
            self.llm = llm
        else:
            if (engine in OS_MODELS):
                self.llm = OSModel(
                    engine = self.engine,
                    **kwargs
                )
            elif (engine in OPENAI_MODELS):
                self.llm = LLMModel(
                    engine = self.engine,
                    **kwargs
                )
            else:
                raise ValueError(f"Invalid engine {engine}")
        if (task not in TASKS):
            raise ValueError(f"Invalid task {task}")
        if (prompt not in PROMPTS):
            raise ValueError(f"Invalid prompt {prompt}")
        if data_dir is None:
            if (task == 'gsm_baseline'):
                data_dir = 'src/gsm_data'
            elif (task == 'entailment_baseline'):
                data_dir = 'src/entailment_data/baseline_data'
        self.save_dir = save_dir
        self.results_filepaths = []
    def run(self):
        pass
    def parse_answers(self):
        pass
    def grade_answer(self):
        pass

def parse_problem(problem_dict, task):
    if task not in TASKS:
        raise ValueError(f"Invalid task {task}")
    if (task == 'gsm_baseline'):
        return problem_dict['input']
    elif (task == 'entailment_baseline'):
        p = 'Hypothesis: ' + problem_dict['hypothesis'] + '\n\n' + 'Text:'
        for sent, text in problem_dict['meta']['triples'].items():
            p += '\n' + sent + ': ' + text
    











async def async_generate_answers(llm, prompt_template, problems):
  '''Generate the answer for the given problem.'''
  outputs = await llm([prompt_template.format(context = prob) for prob in problems])
  return outputs

async def run_model(llm, prompt_template, data, output_file = None):
    problems = [d['input'] for d in data]
    step = 5
    for i in range(0, len(problems), step):
        outputs = await async_generate_answers(llm, prompt_template, problems[i:min(i + step, len(problems))])
        if (i == 0):
            print(outputs[0])
        for j in range(step):
            if (i + j) < len(problems):
                data[i + j]['output'] = outputs[j]
        if (output_file is not None):
            with open(output_file, 'w') as f:
                f.write(json.dumps(data) + '\n')
        print (f"Completed {i + step} problems")
    return data

async def run_baseline(llm, task, prompt_technique, model_name, data_dir, save_dir):
    '''
        Args:
            llm: the language model,
            task: the task to run (e.g. gsm_baseline),
            prompt_technique: the prompt technique to use (e.g. 4cot_gsm),
            model_name: the name of the model,
            data_dir: the directory where the data is stored,
            save_dir: the directory where the output should be saved
    '''
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
            problems = await run_model(llm, prompt_template, data)
            output_file = os.path.join(save_dir, f'gsmic_mixed_{i}_{variant}_output_{model_name}_{prompt_technique}.json')
            print(output_file)
            with open(output_file, 'w') as f:
                  f.write(json.dumps(problems) + '\n')


def grade(filepath, overwrite = False):
        with open(filepath, 'r') as f:
            problems = json.load(f)
        for p in problems:
            if (p['answer'] == 'undefined'):
                p['correct'] = False
            elif (check_corr(p['answer'], p['target'])):
                p['correct'] = True
            else:
                p['correct'] = False
        if not overwrite:
            file_path = os.path.join(os.path.dirname(filepath), 'graded_' + os.path.basename(filepath))
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
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

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
    # if there is a line with def solution(), only take the lines with and after the function
    for i, line in enumerate(answer):
        if (line.lower().find('def solution') != -1):
            answer = answer[i:]
            break
    # if there is no line with def solution(), add one to beginning
    if (answer[0].lower().find('def solution') == -1):
        answer = ['def solution():'] + answer
    # if there is a return statement, replace with print statement
    for i, line in enumerate(answer):
        if (line.lower().find('return') != -1):
            result = line.split('return')[1]
            answer[i] = line.split('return')[0] + f'print({result})'
            break
    # Remove lines after print statement
    for i, line in enumerate(answer):
        if (line.lower().find('print') != -1):
            answer = answer[:i + 1]
            break
    # add a line to run solution()
    answer.append('solution()')
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
    elif task == 'entailment': 
        if (answer.find('Entailment Tree:') == -1):
            return "undefined"
        else:
            answer = answer[answer.find('Entailment Tree:'):]
        answer = get_entailment_proof([answer])[0]
        # Split by lines, only take lines including and after 'Entailment Tree:'
        return answer
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

def to_tsv(filepath, lines):
    with open(filepath, 'w') as f:
        for line in lines:
            f.write(line + '\n')
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
