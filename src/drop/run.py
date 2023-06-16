import os
import sys
import json
import utils
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

# path_root = Path(__file__).parent[2]
# sys.path.append(str(path_root))

import src.drop.feedback as feedback_utils
from src.drop.task_init import DropInit
from src.drop.task_iterate import DropIterate
from src.drop.feedback import (
    CommonsenseFeedback,
    RepetitionFeedback,
    MissingStepFeedback,
    IrrelevancyFeedback,
    RedundancyFeedback,
    SelfRefineFeedback,
)
from src.utils import OPENAI_ENGINES, OS_ENGINES
from src.utils import (
    FeedbackFactory,
    Logger,
    parse_feedback,
    retry_parse_fail_prone_cmd,
)


@retry_parse_fail_prone_cmd
def iterative_drop(
    question: Dict[str, str],
    max_attempts: int,
    temperature: float,
    engine: str,
    feedback_types: List[str],
):
    task_init = DropInit(
        engine=engine,
        prompt_examples="prompt/drop_maf/init.txt",
        temperature=temperature,
    )
    missing_step = MissingStepFeedback(
        engine=engine,
        prompt_examples="prompt/drop_maf/missing_step.txt",
        temperature=temperature,
    )
    commonsense = CommonsenseFeedback(
        engine=engine,
        prompt_examples="prompt/drop_maf/commonsense.txt",
        temperature=temperature,
    )
    repetition = RepetitionFeedback(
        engine=engine,
        prompt_examples="prompt/drop_maf/repetition.txt",
        temperature=temperature,
    )
    irrelevancy = IrrelevancyFeedback(
        engine=engine,
        prompt_examples="prompt/drop_maf/irrelevancy.txt",
        temperature=temperature,
    )
    redundancy = RedundancyFeedback(
        engine=engine,
        prompt_examples="prompt/drop_maf/redundancy.txt",
        temperature=temperature,
    )
    self_refine = SelfRefineFeedback(
        engine=engine,
        prompt_examples="prompt/drop_maf/self_refine.txt",
        temperature=temperature,
    )
    task_iterate = DropIterate(
        engine=engine,
        prompt_examples="prompt/drop_maf/iterate_short.txt",
        temperature=temperature,
    )

    n_attempts = 0
    log = []
    ms_feedback = ""
    commonsense_feedback = ""
    repetition_feedback = ""
    irrelevancy_feedback = ""
    redundancy_feedback = ""
    self_refine_feedback = ""
    solution = ""

    ms_retry = "missing_step" in feedback_types
    commonsense_retry = "commonsense" in feedback_types
    repetition_retry = "repetition" in feedback_types
    irrelevancy_retry = "irrelevancy" in feedback_types
    redundancy_retry = "redundancy" in feedback_types
    self_refine_retry = "self_refine" in feedback_types

    while n_attempts < max_attempts:
        # solutions is actually just one solution to one question but calling task_init takes in a list of questions and returns a list of solutions so we deal with it
        if n_attempts == 0:
            usage, solutions = task_init(data=[question], concurrent=False)
        solutions_fixed = [{**question, "soln": soln} for soln in solutions]
        if ms_retry:
            usage, ms_feedback = missing_step(
                solutions=solutions_fixed, concurrent=False
            )
            if "it is correct" in ms_feedback[0]["feedback"]:
                ms_retry = False
        if commonsense_retry:
            usage, commonsense_feedback = commonsense(
                solutions=solutions_fixed, concurrent=False
            )
            if "it is correct" in commonsense_feedback[0]["feedback"].lower():
                commonsense_retry = False
        if repetition_retry:
            usage, repetition_feedback = repetition(
                solutions=solutions_fixed, concurrent=False
            )
            if "it is correct" in repetition_feedback[0]["feedback"].lower():
                repetition_retry = False
        if irrelevancy_retry:
            usage, irrelevancy_feedback = irrelevancy(
                solutions=solutions_fixed, concurrent=False
            )
            if "it is correct" in irrelevancy_feedback[0]["feedback"].lower():
                irrelevancy_retry = False
        if redundancy_retry:
            usage, redundancy_feedback = redundancy(
                solutions=solutions_fixed, concurrent=False
            )
            if "it is correct" in redundancy_feedback[0]["feedback"].lower():
                redundancy_retry = False
        if self_refine_retry:
            usage, self_refine_feedback = self_refine(
                solutions=solutions_fixed, concurrent=False
            )
            if "it is correct" in self_refine_feedback[0]["feedback"].lower():
                self_refine_retry = False
        feedback = {
            "Missing Step Feedback": ms_feedback,
            "Commonsense Feedback": commonsense_feedback,
            "Repetition Feedback": repetition_feedback,
            "Irrelevancy Feedback": irrelevancy_feedback,
            "Redundancy Feedback": redundancy_feedback,
            "Self Refine Feedback": self_refine_feedback,
        }

        # remove feedback categories that are not applied
        feedback = {k: v for k, v in feedback.items() if len(v) > 0}
        # with open("tmp/log.txt", "a") as f:
        #     f.write(f"Attempt {n_attempts}\n")
        #     f.write("Question:\n")
        #     f.write(task_init.make_query(question))
        #     f.write("Solution:\n")
        #     f.write(json.dumps(solutions[0], indent=4))
        #     f.write("Feedback:\n")
        #     f.write(json.dumps(feedback, indent=4))
        usage, solutions_fixed = task_iterate(
            solutions=solutions_fixed, feedbacks=feedback, concurrent=False
        )

        log.append(
            {
                "attempt": n_attempts,
                "solution_curr": solutions[0],
                "solution_fixed": solutions_fixed[0],
                "feedback": feedback,
            }
        )
        if not (
            ms_retry
            or commonsense_retry
            or repetition_retry
            or irrelevancy_retry
            or redundancy_retry
            or self_refine_retry
        ):
            break
        solution = solutions_fixed[0]
        solutions = solutions_fixed
        n_attempts += 1
    return log


def fix_drop(
    drop_task_file: str,
    max_attempts: int,
    outfile: str,
    temperature: float,
    engine: str,
    feedback_types: List[str],
):
    drop_questions_df = pd.read_json(drop_task_file, lines=True, orient="records")

    # drop_questions_df = drop_questions_df[:5]
    drop_questions_df["run_logs"] = None
    results = []
    for i, row in tqdm(drop_questions_df.iterrows(), total=len(drop_questions_df)):
        row_copy = row.to_dict()
        try:
            run_logs = iterative_drop(
                question=row_copy,
                max_attempts=max_attempts,
                temperature=temperature,
                engine=engine,
                feedback_types=feedback_types,
            )
            row_copy["run_logs"] = run_logs
            row_copy["generated_answer_ours"] = run_logs[-1]["solution_fixed"]
            row_copy["generated_answer_direct"] = run_logs[0]["solution_curr"]
            results.append(row_copy)
            # if i % 10 == 0:
            #     pd.DataFrame(results).to_json(
            #         outfile + f".{i}.jsonl", orient="records", lines=True
            #     )
        except Exception as e:
            raise e
            pass
    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/drop_",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=1, help="maximum number of attempts"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/drop",
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
    args.feedback_types = args.feedback_types.split(",")
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
    with open("tmp/log.txt", "w") as f:
        f.write("")
    debug_drop = "tmp/debug_drop.jsonl"
    with open(debug_drop, "w") as fout:
        fout.write("")
    examples = [
        {
            "hypothesis": "isaac newton discovered that gravity causes the planets in the solar system to orbit the sun",
            "text": [
                "the sun is a kind of star",
                "planets in the solar system orbit the sun",
                "planets orbit stars",
                "isaac newton discovered the theory of gravity",
                "gravity causes orbits",
            ],
        },
        {
            "hypothesis": "grease is used to decrease the friction on the wheels and gears moving on other surfaces",
            "text": [
                "as the smoothness of something increases , the friction of that something will decrease",
                "grease is used to make an object 's surface more smooth",
                "friction occurs when two object 's surfaces move against each other",
                "wheels / gears usually move against other surfaces",
                "a wheel is a kind of object",
                "a gear is a kind of object",
            ],
        },
    ]
    with open(debug_drop, "a") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")
    logs = fix_drop(
        drop_task_file=debug_drop,
        max_attempts=3,
        outfile="tmp/debug_drop_res_sr.json",
        temperature=0.7,
        engine="text-davinci-003",
        feedback_types=["self_refine"],
    )
    logs = fix_drop(
        drop_task_file=debug_drop,
        max_attempts=3,
        outfile="tmp/debug_drop_res_maf.json",
        temperature=0.7,
        engine="text-davinci-003",
        feedback_types=["missing_step", "repetition"],
    )


def main():
    args = parse_args()


if __name__ == "__main__":
    test()
