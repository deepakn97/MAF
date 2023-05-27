import argparse
from src.baselines.baseline_utils import BaselineWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='text-davinci-003', help='Model name to use (e.g. gpt-3.5-turbo, text-davinci-003, vicuna, etc.)')
    parser.add_argument('-t', '--task', type=str, default='gsm_baseline', help='Task to run (e.g. gsm_baseline, entailment_baseline, etc.)')
    parser.add_argument('-p', '--prompt', type=str, default='pot_gsm', help='Prompt technique to use for the model')
    return parser.parse_args()

def main():
    baseline = BaselineWrapper(args.model, args.task, args.prompt)
    baseline.run()

if __name__ == '__main__':
    args = parse_args()
    main()