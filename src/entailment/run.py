import os
import sys
import json
import utils
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List
import pandas as pd

path_root = Path(__file__).parent[2]
sys.path.append(str(path_root))

import src.entailment.feedback as feedback_utils
from src.entailment.task_init import EntailmentInit
from src.entailment.task_iterate import EntailmentIterate
from src.entailment.feedback import (
    CommonsenseFeedback,
    RepetitionFeedback,
    MissingStepFeedback,
    IrrelevancyFeedback,
    RedundancyFeedback,
)
from src.utils import OPENAI_ENGINES, OS_ENGINES
from src.utils import (
    FeedbackFactory,
    Logger,
    parse_feedback,
    retry_parse_fail_prone_cmd,
)


@retry_parse_fail_prone_cmd
def iterative_entailment(
    question: str, max_attempts: int, temperature: float, engine: str
):
    task_init = EntailmentInit(
        engine=engine,
        prompt_examples="prompt/entailment_maf/init.txt",
        temperature=temperature,
    )
    missing_step = MissingStepFeedback(
        engine=engine,
        prompt_examples="prompt/entailment_maf/missing_step.txt",
        temperature=temperature,
    )
    commonsense = CommonsenseFeedback(
        engine=engine,
        prompt_examples="prompt/entailment_maf/commonsense.txt",
        temperature=temperature,
    )
    repetition = RepetitionFeedback(
        engine=engine,
        prompt_examples="prompt/entailment_maf/repetition.txt",
        temperature=temperature,
    )
    irrelevancy = IrrelevancyFeedback(
        engine=engine,
        prompt_examples="prompt/entailment_maf/irrelevancy.txt",
        temperature=temperature,
    )
    redundancy = RedundancyFeedback(
        engine=engine,
        prompt_examples="prompt/entailment_maf/redundancy.txt",
        temperature=temperature,
    )

    task_iterate = EntailmentIterate(
        engine=engine,
        prompt_examples="prompt/entailment_maf/iterate.txt",
        temperature=temperature,
    )

    n_attempts = 0
    log = []
    ms_feedback = ""
    commonsense_feedback = ""
    repetition_feedback = ""
    irrelevancy_feedback = ""
    redundancy_feedback = ""
    feedback = {
        "Missing Step Feedback": ms_feedback,
        "Commonsense Feedback": commonsense_feedback,
        "Repetition Feedback": repetition_feedback,
        "Irrelevancy Feedback": irrelevancy_feedback,
        "Redundancy Feedback": redundancy_feedback,
    }
    solution = ""
    ms_retry = True
    commonsense_retry = True
    repetition_retry = True
    irrelevancy_retry = True
    redundancy_retry = True

    while n_attempts < max_attempts:
        if n_attempts == 0:
            usage, solution = task_init(solutions=[question], concurrent=False)
        solution_fixed = [soln for soln in solution]
        if ms_retry:
            usage, ms_feedback = missing_step(
                solutions=[solution_fixed], concurrent=False
            )
            if "it is correct" in ms_feedback[0]["feedback"]:
                ms_retry = False
        if commonsense_retry:
            usage, commonsense_feedback = commonsense(
                solutions=[solution_fixed], concurrent=False
            )
            if "it is correct" in commonsense_feedback[0]["feedback"]:
                commonsense_retry = False
        if repetition_retry:
            usage, repetition_feedback = repetition(
                solutions=[solution_fixed], concurrent=False
            )
            if "it is correct" in repetition_feedback[0]["feedback"]:
                repetition_retry = False
        if irrelevancy_retry:
            usage, irrelevancy_feedback = irrelevancy(
                solutions=[solution_fixed], concurrent=False
            )
            if "it is correct" in irrelevancy_feedback[0]["feedback"]:
                irrelevancy_retry = False
        if redundancy_retry:
            usage, redundancy_feedback = redundancy(
                solutions=[solution_fixed], concurrent=False
            )
            if "it is correct" in redundancy_feedback[0]["feedback"]:
                redundancy_retry = False
        feedback = {
            "Missing Step Feedback": ms_feedback,
            "Commonsense Feedback": commonsense_feedback,
            "Repetition Feedback": repetition_feedback,
            "Irrelevancy Feedback": irrelevancy_feedback,
            "Redundancy Feedback": redundancy_feedback,
        }

        usage, solution_fixed = task_iterate(
            solutions=[solution_fixed], feedback=feedback, concurrent=False
        )

        log.append(
            {
                "attempt": n_attempts,
                "solution_curr": solution[0],
                "solution_fixed": solution_fixed[0],
                "feedback": feedback,
            }
        )

        if not (
            ms_retry
            or commonsense_retry
            or repetition_retry
            or irrelevancy_retry
            or redundancy_retry
        ):
            break
        solution = solution_fixed
        n_attempts += 1


def fix_entailment(
    entailment_task_file: str,
    max_attempts: int,
    outfile: str,
    temperature: float,
    engine: str,
):
    entailment_questions_df = pd.read_json(
        entailment_task_file, lines=True, orient="records"
    )
    # entailment_questions_df = entailment_questions_df[:5]
    entailment_questions_df["run_logs"] = None
    results = []
    for i, row in tqdm(
        entailment_questions_df.iterrows(), total=len(entailment_questions_df)
    ):
        row_copy = row.to_dict()
        try:
            run_logs = iterative_entailment(
                question=row["input"],
                max_attempts=max_attempts,
                temperature=temperature,
                engine=engine,
            )
            row_copy["run_logs"] = run_logs
            row_copy["generated_answer_ours"] = run_logs[-1]["solution_fixed"]
            row_copy["generated_answer_direct"] = run_logs[0]["solution_curr"]
            results.append(row_copy)
            if i % 10 == 0:
                pd.DataFrame(results).to_json(
                    outfile + f".{i}.jsonl", orient="records", lines=True
                )
        except Exception as e:
            raise e
            pass
    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    return results

    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/entailment_",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=1, help="maximum number of attempts"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/entailment",
        help="location to save the model",
    )
    parser.add_argument("--feedback_types", type=str, default="")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature for sampling"
    )
    parser.add_argument(
        "--summarize_fb", action="store_true", help="summarize feedback"
    )
    parser.add_argument
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--exp_label", type=str, default="default", help="label for the experiment"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="text-davinci-003",
        choices=OPENAI_ENGINES + OS_ENGINES,
        help="engine to use",
    )
    parser.add_argument("--gpus", type=str, default="0,1", help="gpus to use")
    parser.add_argument("--batch_size", type=int, default=5, help="batch size")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()
    args.outdir = os.path.join(
        args.save_dir, f"{args.exp_label}.temp_{args.temperature}.engine_{args.engine}"
    )
    print(args.outdir)

    os.makedirs(args.outdir, exist_ok=True)
    args.outfile = os.path.join(args.outdir, "results.jsonl")
    # print and save the args
    _logger = utils.Logger(os.path.join(args.outdir, "args.txt"))

    print("=====Input Arguments=====")
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args


def test():
    pass


if __name__ == "__main__":
    args = parse_args()
    logger = utils.Logger(os.path.join(args.outdir, "log.txt"))
    fix_entailment(args)
