import os
os.chdir('/home/d_wang/nlp/MAF')
import sys
sys.path.append('/home/d_wang/nlp/MAF')
import gc
import json
from time import sleep
import nltk
import numpy as np
import argparse
from langchain.llms import OpenAI

from dotenv import load_dotenv
from types import SimpleNamespace
import asyncio
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
from src.baselines.baseline_utils import OSModel, async_generate_answers, gsm_run, gsm_baseline, grade, parse_program_answer, load_gsm_data, parse_program, check_corr, create_prompt_template
import torch
import argparse




def grade_programs(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        data[i]['code'] = parse_program(data[i]['output'])
        data[i]['answer'] = parse_program_answer(data[i]['output'])
        data[i]['correct'] = check_corr(data[i]['target'], data[i]['answer'])
    print('accuracy: ', sum([1 for d in data if d['correct']]) / len(data))
    with open(filepath, 'w') as f:
        json.dump(data, f)
    return data
        

async def main():
    print(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', type=str, default="0, 1", help="CUDA_VISIBLE_DEVICES for the model")
    parser.add_argument('-p', '--prompt_examples', type=str, default="pot_gsm", help="filepath to prompt examples")
    args = parser.parse_args()
    llm = OSModel(
        engine="vicuna",
        temperature = 0.001,
        cuda_visible_devices=args.gpus,
    )
    og_data = load_gsm_data('data/gsm_data/gsmic_mixed_0_original.jsonl')[:1]
    irc_data = load_gsm_data('data/gsm_data/gsmic_mixed_0_irc.jsonl')[:0]
    prompt_template = create_prompt_template('gsm_baseline', 'vicuna', args.prompt_examples)
    og_data = await gsm_run(prompt_template, llm, og_data)
    print(og_data)
    with open(f'output_{args.prompt_examples}.json', 'w') as f:
        f.write(json.dumps(og_data))
    grade_programs(f'output_{args.prompt_examples}.json')

if __name__ == "__main__":
    asyncio.run(main())