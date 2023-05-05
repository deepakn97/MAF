import sys
import time
from typing import Dict, List
from src.utils import Prompt

from prompt_lib.backends import openai_api

class GSMIterate(Prompt):
  def __init__(self, engine: str, prompt_examples: str, temperature: float, max_tokens: int = 300) -> None:
    super().__init__(
      question_prefix="",
      answer_prefix="",
      intra_example_sep="\n\n",
      inter_example_sep="\n\n",
    )
    self.engine = engine
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.instruction = "# Given the feedback and the original code, let's rewrite the code to incorporate all of the feedback. Don't change anything unless it is mentioned in the feedback."
    self.setup_prompt_from_examples_file(prompt_examples)
  
  def setup_prompt_from_examples_file(self, examples_path: str) -> str:
    with open(examples_path, "r") as f:
      self.prompt = f.read()
  
  def __call__(self, solution: str, feedback: Dict[str, str]) -> str:
    generation_query = self.make_query(solution=solution, feedback=feedback)
    # print(generation_query)
    success = False
    while not success:
      try:
        output = openai_api.OpenaiAPIWrapper.call(
          prompt=generation_query,
          engine=self.engine,
          max_tokens=self.max_tokens,
          stop_token="### END ###",
          temperature=self.temperature
        )
        success = True
      except Exception as e:
        success = False
        print(e)
        time.sleep(60)

    entire_output = openai_api.OpenaiAPIWrapper.get_first_response(output)
    # print(f"Iterate Output: {entire_output}")
    if "### END ###" in entire_output:
      entire_output = entire_output.split("### END ###")[0].strip()
    solution = ""
    if "def solution():" in entire_output:
      solution = entire_output.split("def solution():")[1]
      solution = "def solution():" + solution.rstrip()
    return solution
  
  def make_query(self, solution: str, feedback: Dict[str, str]) -> str:
    solution = f"""{solution}{self.intra_example_sep}"""
    for feedback_type, feedback_text in feedback.items():
      solution += f"""{feedback_type}:\n{feedback_text}{self.intra_example_sep}"""
    query = f"{self.prompt}{self.intra_example_sep}{solution}"
    return query
  
def test():
  task_iterate = GSMIterate(
    engine="text-davinci-003",
    prompt_examples="prompt/gsm_maf/iterate.txt",
    temperature=0.7
  )

  wrong_soln = """def solution():
    \"\"\"Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?\"\"\"
    chips_per_square_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
    chips_needed = height * chips_per_square_inch
    chips_available = bags * chips_per_bag
    chips_left = chips_available - chips_needed
    length = chips_left / chips_per_square_inch
    result = length
    return result"""
  feedback = {
    "Missing Step Feedback": """# Let us go through the code step-by-step
    chips_per_square_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
# looks good

# Let's check other parts
    chips_needed = height * chips_per_square_inch
    chips_available = bags * chips_per_bag
# wrong! we need to caclulate the area of the mosaic which can be made by available chips. This can be calculated by dividing chips available with chips per square inch. Let's add it!

# Let's check other parts
    chips_left = chips_available - chips_needed
    length = chips_left / chips_per_square_inch
    result = length
    return result
# looks good""",
    "Logical Reasoning Feedback": """# Let us go through the code step-by-step
    chips_per_square_inch = 12
    chips_per_bag = 72
    bags = 2
    height = 3
# looks good

# Let's check other parts
    chips_needed = height * chips_per_square_inch
    chips_available = bags * chips_per_bag
# wrong! chips needed doesn't make sense here. remove it

# Let's check other parts
    chips_left = chips_available - chips_needed
    length = chips_left / chips_per_square_inch
    result = length
    return result
# wrong! chips_left doesn't make sense here. remove it
# wrong! we want to divide the *area of the mosaic* by the height of the mosaic to get the length of the mosaic. Let's fix it!""",
  }
  print(task_iterate(wrong_soln, feedback))

if __name__ == "__main__":
  test()