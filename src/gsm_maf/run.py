import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.gsm_maf.task_init import GSMInit
from src.gsm_maf.feedback import VariableNameFeedback, MissingStepFeedback, LogicalFeedback
from src.gsm_maf.task_iterate import GSMIterate
from src.utils import retry_parse_fail_prone_cmd, FeedbackFactory

CODEX = "code-davinci-002"
GPT3 = "text-davinci-002"
GPT35 = "text-davinci-003"
GPT3TURBO = "gpt-3.5-turbo"
ENGINE = GPT35


@retry_parse_fail_prone_cmd
def iterative_gsm(question: str, max_attempts: int, temperature: float, engine: str):

    # initialize all the required components

    # generation of the first fast version
    task_init = GSMInit(engine=engine, prompt_examples="prompt/gsm_maf/init.txt", temperature=temperature, max_tokens = 300)

    # getting feedback
    variable_name = VariableNameFeedback(engine=engine, prompt_examples="prompt/gsm_maf/variable_naming.txt", temperature=0.7, max_tokens = 600, answer_prefix="def solution():")

    missing_step = MissingStepFeedback(engine=engine, prompt_examples="prompt/gsm_maf/missing_step.txt", temperature=0.7, max_tokens = 300, answer_prefix="def solution():")

    logical = LogicalFeedback(engine=engine, prompt_examples="prompt/gsm_maf/logical.txt", temperature=0.7, max_tokens = 300, answer_prefix="def solution():")

    task_iterate = GSMIterate(engine=engine, prompt_examples="prompt/gsm_maf/iterate.txt", temperature=temperature, max_tokens = 300)

    n_attempts = 0

    log = []
    ms_feedback = ""
    logical_feedback = ""
    feedback = {"Missing Step Feedback": ms_feedback, "Logical Reasoning Feedback": logical_feedback}
    vn_feedback_and_soln = {"feedback": "", "solution": ""}
    solution = ""
    ms_retry = True
    logical_retry = True
    vn_retry = True

    while n_attempts < max_attempts:

        if n_attempts == 0:
            solution = task_init(solution=question)

        solution_fixed = solution

        if vn_retry:
            vn_feedback_and_soln = variable_name(solution=solution)
            solution_fixed = vn_feedback_and_soln["solution"]
            if "it is correct" in vn_feedback_and_soln["feedback"]:
                vn_retry = False

        if ms_retry:
            ms_feedback = missing_step(solution=solution_fixed)
            if "it is correct" in ms_feedback:
                ms_retry = False

        if logical_retry:
            logical_feedback = logical(solution=solution_fixed)
            if "it is correct" in logical_feedback:
                logical_retry = False

        
        feedback = {"Missing Step Feedback": ms_feedback, "Logical Reasoning Feedback": logical_feedback}

        solution_fixed = task_iterate(solution=solution_fixed, feedback=feedback)

        log.append({"attempt": n_attempts, "solution_curr": solution, "solution_fixed": solution_fixed, "feedback": feedback, "variable_name_feedback": vn_feedback_and_soln["feedback"]})

        if not (ms_retry or logical_retry or vn_retry):
            break

        solution = solution_fixed
        # print(f"Solution at attempt {n_attempts}:\n{solution}\n\n"

        n_attempts += 1

    return log


def fix_gsm(gsm_task_file: str, max_attempts: int, outfile: str, temperature: float, engine: str):


    slow_programs_df = pd.read_json(gsm_task_file, lines=True, orient="records")
    # slow_programs_df = slow_programs_df[:5]
    slow_programs_df["run_logs"] = None
    results = []
    for i, row in tqdm(slow_programs_df.iterrows(), total=len(slow_programs_df)):
        row_copy = row.to_dict()
        try:
            run_logs = iterative_gsm(question=row["input"], max_attempts=max_attempts, temperature=temperature, engine=engine)
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
        fout.write(json.dumps({"input": "Kelly is grocery shopping at a supermarket and is making sure she has enough in her budget for the items in her cart. Her 5 packs of bacon cost $10 in total and she has 6 packets of chicken which each cost twice as much as a pack of bacon. She also has 3 packs of strawberries, priced at $4 each, and 7 packs of apples, each priced at half the price of a pack of strawberries. If Kelly’s budget is $65 then how much money, in dollars, does she have left in her budget?"}))
        
    logs = fix_gsm(
        gsm_task_file="/tmp/debug_gsm.jsonl", max_attempts=3, outfile="/tmp/test.jsonl", temperature=0.7, engine="text-davinci-003"
    )
    for i, log in enumerate(logs):
        print(log["generated_answer_ours"])
        print(log["generated_answer_direct"])


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "test":
        test()
    else:
        import argparse
        args = argparse.ArgumentParser()
        args.add_argument("--gsm_task_file", type=str, default="data/tasks/gsm/gsm.jsonl")
        args.add_argument("--max_attempts", type=int, default=4)
        args.add_argument("--outfile", type=str, default="data/tasks/gsm/gsm_outputs.jsonl")
        args.add_argument("--feedback_types", type=str, default="missing_step, variable_naming, logical, coherency, hallucination")
        args.add_argument("--temperature", type=float, default=0.0)
        args.add_argument("--engine", type=str, default=ENGINE, choices=[CODEX, GPT3, GPT35, GPT3TURBO])
        args = args.parse_args()
        args.outfile = f"{args.outfile}.temp_{args.temperature}.engine_{args.engine}.jsonl"
        fix_gsm(gsm_task_file=args.gsm_task_file, max_attempts=args.max_attempts, outfile=args.outfile, temperature=args.temperature, engine=args.engine)