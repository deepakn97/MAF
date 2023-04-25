import pandas as pd
from tqdm import tqdm


from src.gsm_nl.task_init import GSMInit
from src.gsm_nl.feedback import GSMFeedback
from src.gsm_nl.task_iterate import GSMIterate

from src.utils import retry_parse_fail_prone_cmd

CODEX = "code-davinci-002"
GPT3 = "text-davinci-003"
GPT3TURBO = "gpt-3.5-turbo"
ENGINE = GPT3


@retry_parse_fail_prone_cmd
def iterative_gsm(question: str, max_attempts: int, feedback_type: str, temperature: float):

    # initialize all the required components

    # generation of the first fast version
    task_init = GSMInit(engine=ENGINE, prompt_examples="data/prompt/gsm_nl/init.txt", temperature=temperature, max_tokens=500)

    # getting feedback
    if feedback_type == "naive":
        raise NotImplementedError
    else:
        task_feedback = GSMFeedback(engine=ENGINE, prompt_examples="data/prompt/gsm_nl/feedback.txt", temperature=0.7, max_tokens=900)
    
    # run the iterative process
    task_iterative = GSMIterate(engine=ENGINE, prompt_examples="data/prompt/gsm_nl/iterate.txt", temperature=temperature, max_tokens=500)


    n_attempts = 0

    log = []
    solution = ""
    feedback = ""

    while n_attempts < max_attempts:

        if n_attempts == 0:
            solution = task_init(question=question)
        else:
            solution = task_iterative(question=question, solution=solution, feedback=feedback)

        feedback = task_feedback(question=question, solution=solution)
        

        log.append({"attempt": n_attempts, "solution_curr": solution, "feedback": feedback})

        if "incorrect" not in feedback.lower():
            break

        n_attempts += 1

    return log


def fix_gsm(gsm_task_file: str, max_attempts: int, outfile: str, feedback_type: str, temperature: float):


    slow_programs_df = pd.read_json(gsm_task_file, lines=True, orient="records")
    # slow_programs_df = slow_programs_df[:5]
    slow_programs_df["run_logs"] = None
    results = []
    for i, row in tqdm(slow_programs_df.iterrows(), total=len(slow_programs_df)):
        row_copy = row.to_dict()
        try:
            run_logs = iterative_gsm(question=row["input"], max_attempts=max_attempts, feedback_type=feedback_type, temperature=temperature)
            row_copy["run_logs"] = run_logs
            row_copy["generated_answer_ours"] = run_logs[-1]["solution_curr"]
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

    
    with open("/tmp/debug_gsm_nl.jsonl", "w") as fout:
        fout.write(json.dumps({"input": "The educational shop is selling notebooks for $1.50 each and a ballpen at $0.5 each.  William bought five notebooks and a ballpen. How much did he spend in all?"}))
        
    logs = fix_gsm(
        gsm_task_file="/tmp/debug_gsm_nl.jsonl", max_attempts=3, outfile="/tmp/test.jsonl", feedback_type="rich", temperature=0.7
    )
    for i, log in enumerate(logs):
        print(f"Improved Generation: {log['generated_answer_ours']}")
        print(f"First Generation: {log['generated_answer_direct']}")


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
        args.add_argument("--feedback_type", type=str, default="rich")
        args.add_argument("--temperature", type=float, default=0.0)
        args = args.parse_args()
        args.outfile = f"{args.outfile}.fb_{args.feedback_type}.temp_{args.temperature}.engine_{ENGINE}.jsonl"
        fix_gsm(gsm_task_file=args.gsm_task_file, max_attempts=args.max_attempts, outfile=args.outfile, feedback_type=args.feedback_type, temperature=args.temperature)