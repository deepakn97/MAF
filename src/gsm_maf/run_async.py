import os
import sys
import json
import utils
import argparse
from typing import Callable, Dict, List
import pandas as pd
from tqdm import tqdm
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

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

    log = []
    feedbacks_refine = {}
    feedbacks = {}
    for name, fm in feedback_modules.items():
        if fm.eager_refine:
            feedbacks_refine[fm.name] = [""] * len(questions)
        else:
            feedbacks[fm.name] = [""] * len(questions)
    solutions = [""] * len(questions)
    solutions_fixed = [""] * len(questions)
    feedbacks_retry = [[True] * len(questions)] * len(feedback_modules)

    while n_attempts < max_attempts:
        if n_attempts == 0:
            solutions = task_init(solution=questions, batch=True)

        solutions_fixed = solutions
        for i, (fm_name, fm) in enumerate(feedback_modules.items()):
            if any(feedbacks_retry[i]):
                # call the feedback module
                fb_and_maybe_solns = fm(solution=solutions, batch=True)

                # if eager_refine is on, get the solutions and feedbacks
                if fm.eager_refine:
                    for j, fb_and_soln in enumerate(fb_and_maybe_solns):
                        feedbacks_refine[fm_name][j] = fb_and_soln["feedback"]
                        solutions_fixed[j] = fb_and_soln["solution"]
                        if "it is correct" in feedbacks_refine[fm_name][j]:
                            feedbacks_retry[i][j] = False
                else:
                    for j, fb in enumerate(fb_and_maybe_solns):
                        feedbacks[fm_name][j] = fb
                        if "it is correct" in feedbacks[fm_name][j]:
                            feedbacks_retry[i][j] = False
        
        # only call iterate if there is at least one feedback without eager_refine
        if len(feedbacks):
            solutions_fixed = task_iterate(solution=solutions_fixed, feedback=feedbacks, batch=True)

        for i, solution, solution_fixed, feedback, feedback_refine in enumerate(zip(solutions, solutions_fixed, feedbacks, feedbacks_refine)):
            log[i].append({"attempt": n_attempts, "solution_curr": solution, "solution_fixed": solution_fixed, "feedback": feedback, "feedback_with_refinement": feedback_refine})

        if not any(any(feedback_retry) for feedback_retry in feedbacks_retry):
            break

        solutions = solutions_fixed

        n_attempts += 1

    return log


def fix_gsm(gsm_task_file: str, max_attempts: int, outfile: str, temperature: float, feedback_types: str, engine: str):

    # prepare feedback modules
    feedbacks = feedback_types.split(",")
    available_feedbacks= FeedbackFactory.registry.keys()
    feedback_modules = {}
    for feedback in feedbacks:
        if feedback not in available_feedbacks:
            logger.write(f"Feedback {feedback} not found. Available feedbacks are {available_feedbacks}")
        feedback_modules[feedback] = FeedbackFactory.create_feedback(feedback, prompt_examples=f"prompt/gsm_maf/{feedback}.txt", engine=engine, temperature=temperature)

    # generation of the first fast version
    task_init = GSMInit(engine=engine, prompt_examples="prompt/gsm_maf/init.txt", temperature=temperature, max_tokens = 300)

    task_iterate = GSMIterate(engine=engine, prompt_examples="prompt/gsm_maf/iterate.txt", temperature=temperature, max_tokens = 300)

    slow_programs_df = pd.read_json(gsm_task_file, lines=True, orient="records")
    slow_programs_df = slow_programs_df[:5]
    slow_programs_df["run_logs"] = None
    results = []
    # loop over number of attempts instead of number of datapoints to use async calls

    for i, row in tqdm(slow_programs_df.iterrows(), total=len(slow_programs_df)):
        row_copy = row.to_dict()
        try:
            run_logs = iterative_gsm(question=row["input"], max_attempts=max_attempts, temperature=temperature)
            row_copy["run_logs"] = run_logs
            row_copy["generated_answer_ours"] = run_logs[-1]["solution_fixed"]
            row_copy["generated_answer_direct"] = run_logs[0]["solution_curr"]
            results.append(row_copy)
            if i % 10 == 0:
                pd.DataFrame(results).to_json(outfile + f".{i}.jsonl", orient="records", lines=True)
        except Exception as e:
            raise e
            pass
    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    return results


def test():
    import json

    
    with open("/tmp/debug_gsm.jsonl", "w") as fout:
        fout.write(json.dumps({"input": "Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?"}))
        
    logs = fix_gsm(
        gsm_task_file="/tmp/debug_gsm.jsonl", max_attempts=3, outfile="/tmp/test.jsonl", temperature=0.7
    )
    for i, log in enumerate(logs):
        print(log["generated_answer_ours"])
        print(log["generated_answer_direct"])


def parse_args():
    parser = argparse.ArgumentParser()
    args.add_argument("--gsm_task_file", type=str, default="data/tasks/gsm/gsm.jsonl")
    args.add_argument("--max_attempts", type=int, default=4)
    args.add_argument("--save_dir", type=str, default="outputs/gsm_maf")
    args.add_argument("--exp_label", type=str, default="gsmic_mixed_0")
    args.add_argument("--feedback_types", type=str, default="missing_step, variable_naming, logical, coherency, hallucination")
    args.add_argument("--temperature", type=float, default=0.0)
    args.add_argument("--engine", type=str, default=ENGINE, choices=[CODEX, GPT3, GPT35, GPT3TURBO])
    args = parser.parse_args()
    args.outfile = f"{args.save_dir}/{args.exp_label}.temp_{args.temperature}.engine_{args.engine}"

    # print and save the args
    _logger = utils.Logger(f"{args.outfile}.args.txt")

    print('=====Input Arguments=====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        test()
    else:
        args = parse_args()
        logger = utils.Logger(f"{args.outfile}.log.txt")
        fix_gsm(gsm_task_file=args.gsm_task_file, max_attempts=args.max_attempts, outfile=args.outfile, temperature=args.temperature, feedback_types = args.feedback_types, engine=args.engine)