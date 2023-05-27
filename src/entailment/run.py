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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--save_dir', type=str, default='models', help='location to save the model')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--exp_label', type=str, default='default', help='label for the experiment')

    args = parser.parse_args()
    args.ckpt_path = os.path.join(args.save_dir, args.exp_label)

    # print and save the args
    _logger = utils.Logger(os.path.join(args.ckpt_path, 'args.txt'))

    print('=====Input Arguments=====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args

def test():
    pass

def main():
    pass

if __name__ == '__main__':
    args = parse_args()
    logger = utils.Logger(os.path.join(args.ckpt_path, 'log.txt'))
    main()