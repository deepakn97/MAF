import os
import sys
import json
from time import time
import nltk
import numpy as np
import utils
import argparse
from langchain.llms import OpenAI
import baseline_utils
from roscoe.roscoe import score_chains

PROMPT_TEMPLATES = {
  'cot_gsm': {
    'template': "{examples}\nQ: {context}\nA:",
    'input_vars': ['examples', 'context'],
  },
  '0cot_gsm': {
    'template': "Q: {context}\nA: Let's think step by step:",
    'input_vars': ['context']
  },
  'ltm_gsm': {
    'template': "{examples}\nQ: {context}\nA: Let's break down this problem:",
    'input_vars': ['examples', 'context'],
  },
  'pot_gsm': {
    'template': "{examples}\nQ: {context}\nA: ",
    'input_vars': ['examples', 'context'],
  }
}

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data', help='location of the data corpus')
  parser.add_argument('--save_dir', type=str, default='models', help='location to save the model')
  parser.add_argument('--debug', action='store_true', help='debug mode')
  parser.add_argument('--exp_label', type=str, default='default', help='label for the experiment')

  parser.add_argument('--task', type=str, default='cot_gsm', help='task name')
  parser.add_argument('--model', type=str, default='code-davinci-002', help='model name')
  parser.add_argument('--max_tokens', type=int, default=128, help='max tokens for gpt3')
  parser.add_argument('--temperature', type=float, default=0.0, help='temperature for gpt3')

  args = parser.parse_args()
  args.ckpt_path = os.path.join(args.save_dir, args.exp_label)

  # print and save the args
  _logger = utils.Logger(os.path.join(args.ckpt_path, 'args.txt'))

  print('=====Input Arguments=====')
  _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

  return args

def generate_answers(llm, prompt_template, problems):
  '''Generate the answer for the given problem.'''
  for i, prob in enumerate(problems):
    inp = prompt_template.format({'context': prob['context']})
    success = False
    while not success:
      try:
        prob['output'] = llm(inp)
        success = True
      except Exception as e:
        logger.write(f'API server overloaded. Waiting for 30 seconds...')
        time.sleep(30)
        continue

def gsm_baseline():
    prompt_template = utils.create_prompt_template(args.task)
    llm = OpenAI(
        model_name=args.model,
        max_tokens=args.max_tokens,
        stop=['\\n\\n', 'A:', 'Q:'],
        temperature=args.temperature
  ) 
    for i in range(4):
        for variant in ['original', 'irc']:
            data = baseline_utils.load_gsm_data(os.path.join(args.data_dir, f'gsmic_mixed_{i}_{variant}.jsonl'))
            problems = [{'context': d['input'], 'target': d['target']} for d in data]
            generate_answers(llm, prompt_template, problems)
            for problem in problems:
                problem['output'] = problem['output'].split('A: ')[1].split('\n')[0]
                problem['parsed_output'] = baseline_utils.parse_answer(problem['output'])
            with open(os.path.join(args.save_dir, f'gsmic_mixed_{i}_{variant}_output.jsonl'), 'w') as f:
                  for problem in problems:
                        f.write(json.dumps(problem) + '\n')        

        

def main():
  gsm_baseline()

if __name__ == '__main__':
  args = parse_args()
  logger = utils.Logger(os.path.join(args.ckpt_path, 'log.txt'))
  main()