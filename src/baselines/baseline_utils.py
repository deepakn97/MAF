import os
import json
import nltk
import numpy as np
import re
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from io import StringIO
from contextlib import redirect_stdout


def load_gsm_data(filepath):
    '''Loads jsonl file of all questions and associated answers into list of dict objects'''
    with open(filepath, 'r') as read_file:
        data = [json.loads(json_str) for json_str in read_file]
    return data


def create_prompt_template(task):
    with open('prompts.json', 'r') as f:
        PROMPT_TEMPLATES = json.load(f)['prompts']
        template = PROMPT_TEMPLATES[task]
        prompt = PromptTemplate(
            input_variables=template['input_vars'], template=template['template'])

    return prompt


def parse_answer(answer, task):
    if task == 'pot_gsm':
        try:
            answer = answer.split('\n')
            # remove line sayiing "Lets solve this with a python program"
            if (answer[0].lower().find('python program') != -1):
                answer = answer[1:]
            # remove lines after print statement
            for i, line in enumerate(answer):
                if (line.lower().find('print') != -1):
                    answer = answer[:i + 1]
                    break
            answer = '\n'.join(answer)
            f = StringIO()
            with redirect_stdout(f):
                exec(answer)
            answer = f.getvalue()
            answer = answer.split('\n')[0]
            answer = ''.join(c for c in answer if c.isdigit() or c == '.')
            return answer
        except:
            return "undefined"
    else:
        answer_key = 'final_answer: '
        if (answer.find(answer_key) == -1):
            return ""
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
