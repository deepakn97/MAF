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

import src.entailment.feedback as feedback_utils
from src.entailment.task_init import EntailmentInit
from src.entailment.task_iterate import EntailmentIterate
from src.entailment.feedback import (
    CommonsenseFeedback,
    RepetitionFeedback,
    MissingStepFeedback,
    RedundancyFeedback,
    SelfRefineFeedback,
)
from src.entailment.utils import extract_eager
from src.utils import OPENAI_ENGINES, OS_ENGINES
from src.utils import (
    FeedbackFactory,
    Logger,
    parse_feedback,
    retry_parse_fail_prone_cmd,
)


@retry_parse_fail_prone_cmd
def iterative_entailment(
    question: Dict[str, str],
    max_attempts: int,
    temperature: float,
    engine: str,
    feedback_types: List[str],
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
    redundancy = RedundancyFeedback(
        engine=engine,
        prompt_examples="prompt/entailment_maf/redundancy.txt",
        temperature=temperature,
    )
    self_refine = SelfRefineFeedback(
        engine=engine,
        prompt_examples="prompt/entailment_maf/self_refine.txt",
        temperature=temperature,
    )
    task_iterate = EntailmentIterate(
        engine=engine,
        prompt_examples="prompt/entailment_maf/iterate_short.txt",
        temperature=temperature,
    )

    n_attempts = 0
    log = []
    ms_feedback = ""
    commonsense_feedback = ""
    repetition_feedback = ""
    redundancy_feedback = ""
    self_refine_feedback = ""
    solution = ""

    ms_retry = "missing_step" in feedback_types
    commonsense_retry = "commonsense" in feedback_types
    repetition_retry = "repetition" in feedback_types
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
            or redundancy_retry
            or self_refine_retry
        ):
            break
        solution = solutions_fixed[0]
        solutions = solutions_fixed
        n_attempts += 1
    return log


def self_refine(
    question: Dict[str, str],
    max_attempts: int,
    temperature: float,
    engine: str,
):
    task_init = EntailmentInit(
        engine=engine,
        prompt_examples="prompt/entailment_maf/init.txt",
        temperature=temperature,
    )
    self_refine = SelfRefineFeedback(
        engine=engine,
        prompt_examples="prompt/entailment_maf/self_refine_loose.txt",
        temperature=temperature,
    )

    n_attempts = 0
    log = []
    self_refine_feedback = ""
    solution = ""
    self_refine_retry = True

    while n_attempts < max_attempts:
        # solutions is actually just one solution to one question but calling task_init takes in a list of questions and returns a list of solutions so we deal with it
        if n_attempts == 0:
            usage, solutions = task_init(data=[question], concurrent=False)
            solutions_sr = solutions
        solutions_fixed = [{**question, "soln": soln} for soln in solutions]
        if self_refine_retry:
            usage, self_refine_feedback = self_refine(
                solutions=solutions_fixed, concurrent=False
            )
            if "it is correct" in self_refine_feedback[0]["feedback"].lower():
                self_refine_retry = False
        feedback = {
            "Self Refine Feedback": self_refine_feedback,
        }
        rewrites = extract_eager([self_refine_feedback[0]["feedback"]])
        solutions_fixed = rewrites

        log.append(
            {
                "attempt": n_attempts,
                "solution_curr": solutions[0],
                "solution_fixed": solutions_fixed[0],
                "feedback": feedback,
            }
        )
        if not (
            self_refine_retry
        ):
            break
        solutions = solutions_fixed
        n_attempts += 1
    return log

def compare_maf_sr(
    question: Dict[str, str],
    max_attempts: int,
    temperature: float,
    engine: str,
    feedback_types: List[str],
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
    redundancy = RedundancyFeedback(
        engine=engine,
        prompt_examples="prompt/entailment_maf/redundancy.txt",
        temperature=temperature,
    )
    self_refine = SelfRefineFeedback(
        engine=engine,
        prompt_examples="prompt/entailment_maf/self_refine_loose.txt",
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
    redundancy_feedback = ""
    self_refine_feedback = ""
    solution = ""

    ms_retry = "missing_step" in feedback_types
    commonsense_retry = "commonsense" in feedback_types
    repetition_retry = "repetition" in feedback_types
    redundancy_retry = "redundancy" in feedback_types
    self_refine_retry = "self_refine" in feedback_types

    while n_attempts < max_attempts:
        # solutions is actually just one solution to one question but calling task_init takes in a list of questions and returns a list of solutions so we deal with it
        if n_attempts == 0:
            usage, solutions = task_init(data=[question], concurrent=False)
            solutions_sr = solutions
        solutions_fixed = [{**question, "soln": soln} for soln in solutions]
        solutions_fixed_sr = [{**question, "soln": soln} for soln in solutions_sr]
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
        if redundancy_retry:
            usage, redundancy_feedback = redundancy(
                solutions=solutions_fixed, concurrent=False
            )
            if "it is correct" in redundancy_feedback[0]["feedback"].lower():
                redundancy_retry = False
        if self_refine_retry:
            # print("SELF_REFINING")
            # print(solutions_fixed_sr)
            usage, self_refine_feedback = self_refine(
                solutions=solutions_fixed_sr, concurrent=False
            )
            if "it is correct" in self_refine_feedback[0]["feedback"].lower():
                self_refine_retry = False
        feedback = {
            "Missing Step Feedback": ms_feedback,
            "Commonsense Feedback": commonsense_feedback,
            "Repetition Feedback": repetition_feedback,
            "Redundancy Feedback": redundancy_feedback,
        }
        sr_feedback = {
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
        rewrites = extract_eager(solutions_fixed)
        solutions_fixed = rewrites
        self_refine_rewrites = extract_eager([f['feedback'] for f in self_refine_feedback])
        solutions_fixed_sr = self_refine_rewrites

        log.append(
            {
                "attempt": n_attempts,
                "solution_curr": solutions[0],
                "solution_fixed": solutions_fixed[0],
                "feedback": feedback,
            }
        )
        log.append(
            {
                "attempt": n_attempts,
                "solution_curr": solutions[0],
                "solution_fixed": solutions_fixed_sr[0],
                "feedback": sr_feedback,
            }
        )
        if not (
            ms_retry
            or commonsense_retry
            or repetition_retry
            or redundancy_retry
            or self_refine_retry
        ):
            break
        solution = solutions_fixed[0]
        solutions_sr = solutions_fixed_sr
        solutions = solutions_fixed
        n_attempts += 1
    return log


def fix_entailment(
    entailment_task_file: str,
    max_attempts: int,
    outfile: str,
    temperature: float,
    engine: str,
    feedback_types: List[str],
):
    entailment_questions_df = pd.read_json(
        entailment_task_file, lines=True, orient="records"
    )
    entailment_questions_df["run_logs"] = None
    results = []
    for i, row in tqdm(
        entailment_questions_df.iterrows(), total=len(entailment_questions_df)
    ):
        row_copy = row.to_dict()
        try:
            # run_logs = iterative_entailment(
            #     question=row_copy,
            #     max_attempts=max_attempts,
            #     temperature=temperature,
            #     engine=engine,
            #     feedback_types=feedback_types,
            # )
            if ('self_refine' in feedback_types):
                run_logs = self_refine(
                    question=row_copy,
                    max_attempts=max_attempts,
                    temperature=temperature,
                    engine=engine,
                )
            else:
                run_logs = iterative_entailment(
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
            if i % 10 == 0:
                pd.DataFrame(results).to_json(
                    '.'.join(outfile.split('.')[:-1]) + f"_{i}.jsonl", orient="records", lines=True
                )
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
    debug_entailment = "tmp/debug_entailment.jsonl"
    with open(debug_entailment, "w") as fout:
        fout.write("")
    examples = [
        {
            "hypothesis": "earth rotating on its axis causes cycles of day and night on earth",
            "text": [
                "the earth rotates on its tilted axis",
                "a planet rotating causes cycles of day and night on that planet",
                "earth is a kind of planet"
            ]
        }, 
        {
            "hypothesis": "venus reflecting more sunlight than other planets make it brighter than other planets",
            "text": [
                "more light gets reflected on highly reflective things",
                "venus is covered in highly reflective clouds",
                "as the light reflected off of an object increases , the object will appear to be brighter"
            ]
        },
        # {
        #     "hypothesis": "the milky way is a kind of spiral galaxy",
        #     "text": [
        #         "spiral is a kind of shape",
        #         "the milky way is spiral in shape",
        #         "galaxies can be classified by shape",
        #         "the milky way is a kind of galaxy"
        #     ]
        # }
        {
            "hypothesis": "new york state has the greatest sunlight during june",
            "text": [
                "the amount of daylight is greatest in the summer",
                "united states is located in the northern hemisphere",
                "new york / new york state is a state located in the united states of america",
                "june is during the summer in the northern hemisphere",
                "the amount of daylight is greatest on the summer solstice",
                "the amount of daylight is least in the winter",
            ],
        },
        {
            "hypothesis": "the mountain used to be covered by oceans in the past",
            "text": [
                "fossils of sharks are found on top of a mountain",
                "sharks live in oceans",
                "a shark is a kind of fish",
                "if fossils of an aquatic animal or plant are found in a place then that place used to be covered by water in the past",
                "the top of a mountain is a kind of place",
                "an ocean is a kind of body of water",
                "a fish is a kind of aquatic animal",
            ],
            "correct_solution": "sent3 & sent7 -> int1: a shark is a kind of aquatic animal; int1 & sent4 -> int2: if fossils of sharks are found in a place then that place used to be covered by water in the past; int2 & sent2 & sent6 -> int3: if fossils of sharks are found in a place then that place used to be covered by oceans in the past; int3 & sent1 & sent5 -> hypothesis; ",
        },
        # {
        #     "hypothesis": "the sun will appear larger than other stars because it is the closest star to earth",
        #     "text": [
        #         "a planet is a kind of celestial object / body",
        #         "as distance from an object decreases , that object will appear larger",
        #         "earth is a kind of planet",
        #         "a star is a kind of celestial object / celestial body",
        #         "the sun is the star that is closest to earth",
        #     ],
        #     "correct_solution": "sent1 & sent3 -> int1: earth is a kind of celestial object; int1 & sent2 & sent4 -> int2: as the distance from a star to earth decreases, the star will appear larger; int2 & sent5 -> hypothesis; ",
        # }
        # {
        #     "hypothesis": "isaac newton discovered that gravity causes the planets in the solar system to orbit the sun",
        #     "text": [
        #         "the sun is a kind of star",
        #         "planets in the solar system orbit the sun",
        #         "planets orbit stars",
        #         "isaac newton discovered the theory of gravity",
        #         "gravity causes orbits",
        #     ],
        #     "correct_solution": "sent3 & sent5 -> int1: gravity causes planets to orbit stars; int1 & sent1 & sent2 -> int2: gravity causes the planets in solar system to orbit the sun; int2 & sent4 -> hypothesis; ",
        # },
        # {
        #     "hypothesis": "grease is used to decrease the friction on the wheels and gears moving on other surfaces",
        #     "text": [
        #         "as the smoothness of something increases , the friction of that something will decrease",
        #         "grease is used to make an object 's surface more smooth",
        #         "friction occurs when two object 's surfaces move against each other",
        #         "wheels / gears usually move against other surfaces",
        #         "a wheel is a kind of object",
        #         "a gear is a kind of object",
        #     ],
        # },
    ]
    with open(debug_entailment, "a") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")
    logs = fix_entailment(
        entailment_task_file=debug_entailment,
        max_attempts=3,
        outfile="tmp/debug_entailment_res.json",
        temperature=0.0,
        engine="text-davinci-003",
        feedback_types=["missing_step", "repetition", "self_refine", "redundancy"],
    )


def run_self_refine(model):
    task_1_dev = "data/entailment_data/baseline_data/task_1_dev.jsonl"
    task_2_dev = "data/entailment_data/baseline_data/task_2_dev.jsonl"
    for i, dev in enumerate([task_1_dev]):
        with open(dev, "r") as fin:
            examples = [json.loads(line) for line in fin]
        for j in range(len(examples)):
            examples[j] = {
                "hypothesis": examples[j]["hypothesis"],
                "text": [v for k, v in examples[j]["meta"]["triples"].items()]
            }
        questions = "tmp/questions.jsonl"
        with open(questions, "w") as fout:
            for example in examples:
                fout.write(json.dumps(example) + "\n")
        logs = fix_entailment(
            entailment_task_file=questions,
            max_attempts=3,
            temperature = 0.0,
            engine=model,
            feedback_types=['self_refine'],
            outfile=f"tmp/self_refine_res_{i}.json",
        )
def main():
    # args = parse_args()
    run_self_refine('text-davinci-003')


if __name__ == "__main__":
    main()
