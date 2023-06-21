import os
import json
import numpy as np

from tqdm import tqdm
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data', help='location of the data corpus')
  parser.add_argument('--data_file', type=str, default=None, help='location of the data file')
  parser.add_argument('--save_dir', type=str, default='models', help='location to save the model')
  parser.add_argument('--outfile_label', type=str, default='0', help='label for the output file')
  parser.add_argument('--sample', action='store_true', help='sample mode')
  parser.add_argument('--num_examples', type=int, default=100, help='number of examples to sample')

  args = parser.parse_args()

  return args

def main():
  # load gsm-ic data
  if args.data_file is None:
    with open(os.path.join(args.data_dir, 'gsmic_2step.json'), 'r') as f:
      two_step_data = json.load(f)
    with open(os.path.join(args.data_dir, 'gsmic_mstep.json'), 'r') as f:
      m_step_data = json.load(f)
    data = two_step_data + m_step_data
    outfile = f'gsmic_mixed_{args.outfile_label}'
  else:
    with open(os.path.join(args.data_dir, f'{args.data_file}.json'), 'r') as f:
      data = json.load(f)
    outfile = f'{args.data_file}_{args.outfile_label}'
  
  if args.sample:
    # shuffle the data and sample num_examples from it
    np.random.shuffle(data)
    data = data[:args.num_examples]

  original_data = []
  irc_data = []
  for dd in tqdm(data):
    original_data.append({'input': dd['original_question'], 'target': dd['answer'], 'n_steps': dd['n_steps']})
    irc_data.append({'input': dd['new_question'], 'target': dd['answer'], 'n_steps': dd['n_steps']})

  
  # save original data
  with open(os.path.join(args.save_dir, f'{outfile}_original.jsonl'), 'w') as f:
    for g in range(len(original_data)):
      f.write(json.dumps(original_data[g]) + '\n')

  # save irc data
  with open(os.path.join(args.save_dir, f'{outfile}_irc.jsonl'), 'w') as f:
    for g in range(len(irc_data)):
      f.write(json.dumps(irc_data[g]) + '\n')

if __name__ == '__main__':
  args = parse_args()
  main()