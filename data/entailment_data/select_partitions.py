import os
import sys
import json
import random
def select_100(infile, outfile):
    print(infile)
    with open(infile, 'r') as f:
        data = [json.loads(json_str) for json_str in f]
    # sample 100 random examples from the data
    data = random.sample(data, 100)

    #clear file
    with open(outfile, 'w') as f:
        f.write('')
    with open(outfile, 'a') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')
    
def main():
    data_folder = '/home/d_wang/nlp/MAF/data/entailment_data/data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset'
    partition_folder = '/home/d_wang/nlp/MAF/data/entailment_data/partitions'
    select_100(os.path.join(data_folder, 'task_1/test.jsonl'), os.path.join(partition_folder, 'task_1.jsonl'))
    select_100(os.path.join(data_folder, 'task_2/test.jsonl'), os.path.join(partition_folder, 'task_2.jsonl'))

if __name__ == '__main__':
    main()