import os
import sys
import json
import utils
import argparse
from pathlib import Path
from typing import List

path_root = Path(__file__).parent[2]
sys.path.append(str(path_root))

import src.entailment.feedback as feedback_utils
from src.entailment.task_init import EntailmentInit
from src.entailment.task_iterate import EntailmentIterate
from src.utils import OPENAI_ENGINES, OS_ENGINES
from src.utils import FeedbackFactory, Logger, parse_feedback

def fix_entailment(args):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/entailment_', help='location of the data corpus')
    parser.add_argument('--max_attempts', type=int, default=1, help='maximum number of attempts')
    parser.add_argument('--save_dir', type=str, default='outputs/entailment', help='location to save the model')
    parser.add_argument('--feedback_types', type=str, default='')
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature for sampling')
    parser.add_argument('--summarize_fb', action='store_true', help='summarize feedback')
    parser.add_argument
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--exp_label', type=str, default='default', help='label for the experiment')
    parser.add_argument('--engine', type=str, default='text-davinci-003', choices=OPENAI_ENGINES + OS_ENGINES, help='engine to use')
    parser.add_argument('--gpus', type=str, default="0,1", help='gpus to use')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    args.outdir = os.path.join(args.save_dir, f"{args.exp_label}.temp_{args.temperature}.engine_{args.engine}")
    print(args.outdir)

    os.makedirs(args.outdir, exist_ok=True)
    args.outfile = os.path.join(args.outdir, 'results.jsonl')
    # print and save the args
    _logger = utils.Logger(os.path.join(args.outdir, 'args.txt'))

    print('=====Input Arguments=====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args

def test():
    pass

if __name__ == '__main__':
    args = parse_args()
    logger = utils.Logger(os.path.join(args.outdir, 'log.txt'))
    fix_entailment(args)