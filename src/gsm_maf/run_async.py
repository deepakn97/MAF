import os
import sys
import json
import time

import numpy as np
import argparse
from typing import Callable, Dict, List
import pandas as pd
from tqdm import tqdm
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import src.gsm_maf.feedback as feedback
from src.gsm_maf.task_init import GSMInit
from src.gsm_maf.task_iterate import GSMIterate
from src.utils import retry_parse_fail_prone_cmd, FeedbackFactory, Logger

CODEX = "code-davinci-002"
GPT3 = "text-davinci-002"
GPT35 = "text-davinci-003"
GPT3TURBO = "gpt-3.5-turbo"
ENGINE = GPT35


@retry_parse_fail_prone_cmd
def iterative_gsm(questions: List[str], max_attempts: int, feedback_modules: Dict[str, Callable], task_init: GSMInit, task_iterate: GSMIterate):

    # initialize all the required components
    n_attempts = 0

    log = [[] for i in range(len(questions))]
    feedbacks_refine = {}
    feedbacks = {}
    for name, fm in feedback_modules.items():
        if fm.eager_refine:
            feedbacks_refine[fm.name] = ["" for i in range(len(questions))]
        else:
            feedbacks[fm.name] = ["" for i in range(len(questions))]
    solutions = ["" for i in range(len(questions))]
    solutions_fixed = ["" for i in range(len(questions))]
    feedbacks_retry = [ [True for i in range(len(questions))] for j in range(len(feedback_modules))]

    while n_attempts < max_attempts:
        # print(feedbacks_retry)
        iter_start = time.time()
        logger.write(f"Running iteration {n_attempts}")
        if n_attempts == 0:
            logger.write("Generating initial solutions\n")
            init_gen_start = time.time()
            usage, solutions = task_init(solutions=questions)
            # print(solutions)
            init_gen_end = time.time()
            mins = (init_gen_end - init_gen_start)/60
            logger.write(f"Initial generation took {mins} minutes\n")
            logger.write(f"Token usage per minute: {usage/mins}")
            time.sleep(60)

        solutions_fixed = solutions
        for i, fm in enumerate(list(feedback_modules.values())):
            logger.write(f"Generating {fm.name}\n")
            logger.write(f"Args for feedback - temperature: {fm.temperature}, max_tokens: {fm.max_tokens}, engine: {fm.engine}\n")
            fb_gen_start = time.time()
            if any(feedbacks_retry[i]):
                # call the feedback module
                retry_idxs = np.where(feedbacks_retry[i])[0]
                solutions_retry = [solutions_fixed[idx] for idx in retry_idxs]
                usage, fb_and_maybe_solns = fm(solutions=solutions_retry)
                # print(fb_and_maybe_solns)

                # if eager_refine is on, get the solutions and feedbacks
                for j, idx in enumerate(retry_idxs):
                    if "it is correct" in fb_and_maybe_solns[j]['feedback']:
                        feedbacks_retry[i][idx] = False
                    if fm.eager_refine:
                        solutions_fixed[idx] = fb_and_maybe_solns[j]["solution"]
                        feedbacks_refine[fm.name][idx] = fb_and_maybe_solns[j]["feedback"]
                    else:
                        feedbacks[fm.name][idx] = fb_and_maybe_solns[j]['feedback']
            fb_gen_end = time.time()
            mins = (fb_gen_end - fb_gen_start)/60
            logger.write(f"{fm.name} generation took {mins} minutes\n")
            logger.write(f"Token usage per minute: {usage/mins}")
            time.sleep(60)
        
        # only call iterate if there is at least one feedback without eager_refine
        if len(feedbacks):
            logger.write("Generating refined solutions\n")
            refine_gen_start = time.time()
            usage, solutions_fixed = task_iterate(solutions=solutions_fixed, feedbacks=feedbacks)
            refine_gen_end = time.time()
            mins = (refine_gen_end - refine_gen_start)/60
            logger.write(f"Refined generation took {mins} minutes\n")
            logger.write(f"Token usage per minute: {usage/mins}")
            time.sleep(60)

        for i in range(len(questions)):
            solution = solutions[i]
            solution_fixed = solutions_fixed[i]
            feedback_with_refinement = {}
            feedback = {}
            for ft, fb in feedbacks.items():
                feedback[ft] = fb[i]
            for ft, fb in feedbacks_refine.items():
                feedback_with_refinement[ft] = fb[i]
            log[i].append({"attempt": n_attempts, "solution_curr": solution, "solution_fixed": solution_fixed, "feedback": feedback, "feedback_with_refinement": feedback_with_refinement})

        if not any(any(feedback_retry) for feedback_retry in feedbacks_retry):
            break

        solutions = solutions_fixed

        n_attempts += 1
        iter_end = time.time()
        logger.write(f"Iteration {n_attempts} took {(iter_end - iter_start)/60}minutes\n")

    return log


def fix_gsm(gsm_task_file: str, max_attempts: int, outfile: str, temperature: float, feedback_types: str, engine: str):

    # prepare feedback modules
    feedbacks = [ft.strip() for ft in feedback_types.split(",")]
    available_feedbacks= list(FeedbackFactory.registry.keys())
    feedback_modules = {}
    for feedback in feedbacks:
        if feedback not in available_feedbacks:
            logger.write(f"Feedback {feedback} not found. Available feedbacks are {available_feedbacks}")
        feedback_modules[feedback] = FeedbackFactory.create_feedback(feedback, prompt_examples=f"prompt/gsm_maf/{feedback}.txt", engine=engine, temperature=temperature)
        # print(feedback_modules[feedback].name)
        # print(feedback_modules[feedback].eager_refine)
        # print(feedback_modules[feedback].max_tokens)

    # generation of the first fast version
    task_init = GSMInit(engine=engine, prompt_examples="prompt/gsm_maf/init.txt", temperature=temperature, max_tokens = 300)

    task_iterate = GSMIterate(engine=engine, prompt_examples="prompt/gsm_maf/iterate.txt", temperature=temperature, max_tokens = 300)

    df = pd.read_json(gsm_task_file, lines=True, orient="records")
    # df = df[:5]
    df["run_logs"] = [None] * len(df)
    results = []
    # loop over number of attempts instead of number of datapoints to use async calls
    run_logs = iterative_gsm(questions=df["input"], max_attempts=max_attempts, feedback_modules=feedback_modules, task_init=task_init, task_iterate=task_iterate)
    for j, row in enumerate(df.iterrows()):
        row_copy = row[-1].to_dict()
        row_copy["run_logs"] = run_logs[j]
        row_copy["generated_answer_direct"] = run_logs[j][0]["solution_curr"]
        row_copy["generated_answer_ours"] = run_logs[j][-1]["solution_fixed"]
        results.append(row_copy)
            
    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    return results

def test():
    import json

    
    with open("/tmp/debug_gsm.jsonl", "w") as fout:
        questions = ["Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?", "Kelly is grocery shopping at a supermarket and is making sure she has enough in her budget for the items in her cart. Her 5 packs of bacon cost $10 in total and she has 6 packets of chicken which each cost twice as much as a pack of bacon. She also has 3 packs of strawberries, priced at $4 each, and 7 packs of apples, each priced at half the price of a pack of strawberries. If Kelly’s budget is $65 then how much money, in dollars, does she have left in her budget?"]
        for q in questions:
            fout.write(json.dumps({"input": q}) + "\n")
        
    logs = fix_gsm(gsm_task_file='/tmp/debug_gsm.jsonl', max_attempts=3, outfile='/tmp/test.jsonl', temperature=0.7, feedback_types='variable_naming, missing_step, logical', engine='text-davinci-003')
    for i, log in enumerate(logs):
        print(log["generated_answer_ours"])
        print(log["generated_answer_direct"])


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--gsm_task_file", type=str, default="data/tasks/gsm/gsm.jsonl")
    args.add_argument("--max_attempts", type=int, default=4)
    args.add_argument("--save_dir", type=str, default="outputs/gsm_maf")
    args.add_argument("--exp_label", type=str, default="gsmic_mixed_0")
    args.add_argument("--feedback_types", type=str, default="variable_naming, missing_step, logical, coherency, hallucination")
    args.add_argument("--temperature", type=float, default=0.0)
    args.add_argument("--engine", type=str, default=ENGINE, choices=[CODEX, GPT3, GPT35, GPT3TURBO])
    args = args.parse_args()
    args.outfile_prefix = f"{args.exp_label}.temp_{args.temperature}.engine_{args.engine}"
    args.outfile = os.path.join(args.save_dir, f"{args.outfile_prefix}.jsonl")    # print and save the args

    _logger = Logger(os.path.join(args.save_dir, f"{args.outfile_prefix}.args.txt"))

    print('=====Input Arguments=====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        logger = Logger(f"/tmp/test.log.txt")
        test()
    else:
        args = parse_args()
        logger = Logger(os.path.join(args.save_dir, f"{args.outfile_prefix}.log.txt"))
        fix_gsm(gsm_task_file=args.gsm_task_file, max_attempts=args.max_attempts, outfile=args.outfile, temperature=args.temperature, feedback_types = args.feedback_types, engine=args.engine)