from importlib import reload
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
import signal
from glob import glob
import re

# from https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration):
  def timeout_handler(signum, frame):
    raise TimeoutError(f"block timedout after {duration} seconds")

  signal.signal(signal.SIGALRM, timeout_handler)
  signal.alarm(duration)
  try:
    yield
  finally:
    signal.alarm(0)

def read_json(path):
  import json
  rows = []
  with open(path, "r") as f:
    for line in f:
      rows.append(json.loads(line))
  
  task_df = pd.DataFrame(rows)
  return task_df

def evaluate_nl_prompt(path, num_gsm: int = 1319):
  data = read_json(path)
  if "question" not in data.columns:
    data["question"] = data["input"]
  if "answer" not in data.columns:
    data["answer"] = data["target"]
  
  attempt_to_acc = []
  num_gsm = len(data)
  reports = []
  for idx, row in tqdm(data.iterrows(), total=len(data)):
    attempt_to_acc_ = {i: 0 for i in range(5)}
    attempt_to_acc_["question"] = row["question"]
    solutions = []
    if row["run_logs"] is None:
      continue
    for _, log in enumerate(row["run_logs"]):
      solutions.append(log["solution_curr"])
    feedback = [rec["feedback"] for rec in row["run_logs"]]

    prev_accuracy = 0
    for iter_idx, soln in enumerate(solutions):
      soln = soln.split("The answer is")
      print(f"Solution: {soln[-1]}")
      if len(soln) == 1:
        soln = ""
        is_corr = 0
        print('No answer found')
      else:
        soln = normalize_answer(soln[1].strip())
        correct_solution = str(row["answer"])
        print(f"Correct solution: {correct_solution}, predicted solution: {soln}")
        is_corr = int(check_corr(soln, correct_solution))

      if iter_idx > 0 and is_corr == 1 and prev_accuracy == 0:
        report = {
           "previous_solution": solutions[iter_idx - 1],
           "feedback": feedback[iter_idx-1],
           "next_solution": solutions[iter_idx]
        }
        reports.append(report)

      if is_corr == 1:
        for i in range(iter_idx, 5):
          attempt_to_acc_[i] = 1
        break
      attempt_to_acc_[iter_idx] = 0
      prev_accuracy = is_corr

    attempt_to_acc.append(attempt_to_acc_)

  df = pd.DataFrame(attempt_to_acc) 

  # print(attempt_to_acc)
  for i in range(5):
    print(f"Accuracy at {i} = {df[i].sum() / num_gsm:.2f} ({df[i].sum()} / {num_gsm})")
  
  df.to_json("/tmp/attempt_to_acc.jsonl", orient="records", lines=True)

  report_file = f"{path}.reports.txt"
  print_reports(reports, report_file)
  return reports


# Step 4
def print_reports(reports, report_file):
  with open(report_file, "w") as f:
    for i, report in enumerate(reports):
      f.write(f"Report {i + 1}:\n")
      f.write("\nPrevious solution:\n")
      f.write(report["previous_solution"])
      f.write("\n\nFeedback:\n")
      f.write(report["feedback"])
      f.write("\n\nNext solution:\n")
      f.write(report["next_solution"])
      f.write("\n\n" + "=" * 80 + "\n\n")

def check_corr(result: float, correct_solution: float, tol: float = 1e-3):
  if result.strip() == correct_solution.strip():
    return 1
  try:
    result = float(result.strip())
    correct_solution = float(correct_solution.strip())
    return abs(result - correct_solution) < tol
  except:
    return 0

def normalize_answer(ans: str) -> str:
  # check for equation
  if "=" in ans:
    ans = ans.split("=")[1].strip()
  
  # extract float/int numbers
  ans = re.findall(r"[-+]?(?:\d*\.*\d+)", ans)
  return ans[-1]


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--path", type=str, default="data/quco/quco_test.jsonl")
  args = parser.parse_args()
  
  evaluate_nl_prompt(args.path)
