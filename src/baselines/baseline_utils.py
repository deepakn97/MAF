import os
import json
import nltk
import numpy as np
import re
from langchain.prompts import FewShotPromptTemplate, PromptTemplate


def prepare_modified_data(problem):
    '''Remove each sentence one by one and see if the model can still predict the label correctly.'''
    sentences = nltk.sent_tokenize(problem['question'])
    modified_data = []
    for i in range(len(sentences)):
        modified_question = '.'.join(sentences[:i] + sentences[i+1:])
        modified_data.append({'context': modified_question,
                             'removed_sentence_id': i, 'removed_sentence': sentences[i]})
    return modified_data


def load_gsm_data(filepath):
    '''Loads jsonl file of all questions and associated answers into list of dict objects'''
    with open(filepath, 'r') as read_file:
        data = [json.loads(json_str) for json_str in read_file]
    return data


def prepare_ic_data(data):
    ic_data = []
    original_data = []
    for d in data:
        ic_data.append({
            'question': d['new_question'],
            'answer': d['answer']
        })
        original_data.append({
            'question': d['original_question'],
            'answer': d['answer']
        })

    return ic_data, original_data


def create_prompt_template(task):
    with open('prompts.json', 'r') as f:
        PROMPT_TEMPLATES = json.load(f)['prompts']
        template = PROMPT_TEMPLATES[task]
        prompt = PromptTemplate(
            input_variables=template['input_vars'], template=template['template'])

    return prompt


def prepare_clean_context(modified_data, scores):
    for md, score in zip(modified_data, scores):
        md['score'] = score


def parse_answer(answer):
    json_start = answer.find('{')
    json_end = answer.find('}') + 1
    if (json_start == -1 or json_end == 0):
        return ""
    json_string = json.loads(answer[json_start:json_end])
    try:
        orig = json_string['final_answer']
        orig = ''.join(c for c in orig if c.isdigit() or c == '.')
        float_orig = float(orig)
        int_orig = int(orig)
        if (float_orig != int_orig):
            return str(float_orig)
        else:
            return str(int_orig)
    except:
        return ""


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
