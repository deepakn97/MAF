import argparse
import json
import os
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.gsm_maf.run import lazy_gsm, eager_gsm
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

# import feedback


def lazy_vs_eager(questions: list[dict], max_attempts: int, temperature: float):
    """
    Each question in questions is expected to have the following fields
    - question: math question being asked
    - answer: correct numerical answer to the question
    """
    count = len(questions)
    lazy_correct = 0
    eager_correct = 0
    total_lazy_logs: list[list[dict]] = []
    total_eager_logs: list[list[dict]] = []
    for question in tqdm(questions):
        query, answer = question['question'], question['answer']

        lazy_logs = lazy_gsm(
            question=query, max_attempts=max_attempts, temperature=temperature)
        lazy_answer = lazy_logs[-1]['solution_fixed']
        lazy_correct += int(lazy_answer == answer)
        total_lazy_logs.append(lazy_logs)

        eager_logs = eager_gsm(
            question=query, max_attempts=max_attempts, temperature=temperature)
        eager_answer = eager_logs[-1]['solution_fixed']
        eager_correct += int(eager_answer == answer)
        total_eager_logs.append(eager_logs)
    # returns tuple(performance, logs)
    return {
        "count": count,
        "lazy_correct": lazy_correct,
        "eager_correct": eager_correct,
        "lazy_logs": total_lazy_logs,
        "eager_logs": total_eager_logs
    }


def iter_vs_base():
    pass


def main():
    pass
