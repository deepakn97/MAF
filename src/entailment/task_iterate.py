import asyncio
from typing import Dict, List

from tqdm import tqdm
from src.utils import Prompt, acall_gpt, call_gpt, extract_answer_gpt

class EntailmentIterate(Prompt):
    def __init__(
        self,
        engine: str,
        prompt_examples: str,
        temperature: float,
        max_tokens: int = 300
    ) -> None:
        super().__init__(
            question_prefix="Hypothesis: ",
            answer_prefix="Entailment Tree:",
            intra_example_sep="\n\n",
            inter_example_sep="### END ###",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.instruction = "# Given the feedback and the original answer, let's rewrite the answer to incorporate all the feedback. Don't change anything unless it is mentioned in the feedback."
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, prompt_examples: str) -> None:
        with open(prompt_examples, "r") as f:
            self.prompt = f.read()

    def make_query(
        self, 
        solution: Dict[str, str],
        feedback: Dict[str, str]
    ) -> str:
        query = f"""{self.question_prefix}{solution['hypothesis']}{self.intra_example_sep}Text:\n"""
        for i, sent in enumerate(solution['text']):
            query += f"# sent {i+1}: {sent}\n"
        
        query += f"""\n{self.answer_prefix}\n{solution['soln']}{self.intra_example_sep}"""
        
        for feedback_type, feedback_text in feedback.items():
            query += f"""{feedback_type}:\n{feedback_text}{self.intra_example_sep}"""
        query = f"""{self.prompt}{self.intra_example_sep}{query}{self.instruction}"""
        return query
    
    def __call__(
        self,
        solutions: List[Dict[str, str]],
        feedbacks: Dict[str, List[str]],
        batch_size: int = 10,
        concurrent: bool = True
    ) -> List[str]:

        generation_queries = []
        for i in range(len(solutions)):
            feedback = {}
            for ft, fb in feedbacks.items():
                feedback[ft] = fb[i]
            generation_queries.append(self.make_query(
                solution=solutions[i],
                feedback=feedback
            ))

        if not concurrent:
            batch_size = 1
        
        responses = []
        for i in tqdm(range(0, len(generation_queries), batch_size), total=len(generation_queries) // batch_size):
            if concurrent:
                batch_responses = asyncio.run(acall_gpt(
                    generation_queries[i:i+batch_size],
                    self.engine,
                    self.temperature,
                    self.max_tokens,
                    stop_token=self.inter_example_sep
                ))
            else:
                batch_responses = call_gpt(
                    generation_queries[i:i+batch_size],
                    self.engine,
                    self.temperature,
                    self.max_tokens,
                    stop_token=self.inter_example_sep
                )
            responses.extend(batch_responses)

        usage, entire_outputs = extract_answer_gpt(responses, self.engine)

        solutions = []
        for entire_output in entire_outputs:
            if self.inter_example_sep in entire_output:
                entire_output = entire_output.split(self.inter_example_sep)[0].strip()
            solution = ""
            if self.answer_prefix in entire_output:
                solution = entire_output.split(self.answer_prefix)[1].strip()

            solutions.append(solution)

        return usage, solutions

def test():
    task_iterate = EntailmentIterate(
        engine="text-davinci-003",
        prompt_examples="prompt/entailment_maf/iterate.txt",
        temperature=0.0
    )

    wrong_solns = [
        {
            "hypothesis": "Evaporating and condensing can both be caused by changes in heat energy",
            "text": [
                "temperature is a measure of heat energy",
                "temperature changes can cause phase changes",
                "evaporating is a kind of phase change",
                "condensing is a kind of phase change"
            ],
            "soln": """sent 1 and sent 2: temperature is a measure of heat energy and temperature changes can cause phase changes -> int 1: changes in heat energy can cause phase changes
sent 2 and sent 3: temperature changes can cause phase changes and evaporating is a kind of phase change -> int 2: temperature changes can cause evaporating
sent 2 and sent 4: temperature changes can cause phase changes and condensing is a kind of phase change -> int 3: temperature changes can cause condensing
sent 1 and int 2 and int 3: temperature is a mesaure of heat energy and temperature changes can cause evaporating and temperature changes can cause condensing -> int 4: changes in heat energy can cause evaporating and condensing
sent 3 and int 1: evaporating is a kind of phase change and changes in heat energy can cause phase changes -> H: Evaporating can be caused by changes in heat energy
sent 4 and int 1: condensing is a kind of phase change and changes in heat energy can cause phase changes -> H: Condensing can be caused by changes in heat energy"""
        },
        {
            "hypothesis": "An astronaut requires the oxygen in a spacesuit backpack to breathe",
            "text": [
                "spacesuit backpacks contain oxygen",
                "an animal requires oxygen to breathe",
                "a vacuum does not contain oxygen",
                "an astronaut is a kind of human",
                "a human is a kind of animal",
                "space is a vacuum"
                "space is cold"
            ],
            "soln": """sent 4 and sent 2: an astronaut is a kind of human and an animal requires oxygen to breathe -> int 1: an astronaut requires oxygen to breathe
sent 1 and int 1: spacesuit backpacks contain oxygen and an astronaut requires oxygen to breathe -> H: An astronaut requires the oxygen in a spacesuit backpack to breathe"""
        }
    ]
    feedbacks = {
        "Missing Step Feedback": ["""# Let us go through the entailment tree line by line.
    sent 1 and sent 2: temperature is a measure of heat energy and temperature changes can cause phase changes -> int 1: changes in heat energy can cause phase changes
# Looks good

# Let's check other parts
    sent 2 and sent 3: temperature changes can cause phase changes and evaporating is a kind of phase change -> int 2: temperature changes can cause evaporating
# Looks good

# Let's check other parts
    sent 2 and sent 4: temperature changes can cause phase changes and condensing is a kind of phase change -> int 3: temperature changes can cause condensing
# Looks good

# Let's check other parts
    sent 1 and int 2 and int 3: temperature is a mesaure of heat energy and temperature changes can cause evaporating and temperature changes can cause condensing -> int 4: changes in heat energy can cause evaporating and condensing
# Looks good

# Let's check other parts
    sent 3 and int 1: evaporating is a kind of phase change and changes in heat energy can cause phase changes -> H: Evaporating can be caused by changes in heat energy
# wrong! The entailment tree only arrives at part of the hypothesis. The sentence 'Evaporating can be caused by changes in heat energy' must be an intermediate.

# Let's check other parts
    sent 4 and int 1: condensing is a kind of phase change and changes in heat energy can cause phase changes -> H: Condensing can be caused by changes in heat energy
# wrong! The entailment tree only arrives at part of the hypothesis. The sentence 'Condensing can be caused by changes in heat energy' must be an intermediate. We can combine this with 'Evaporating can be caused by changes in heat energy' to arrive at the conclusion.""",
    """# Let us go through the entailment tree line by line.
    sent 4 and sent 2: an astronaut is a kind of human and an animal requires oxygen to breathe -> int 1: an astronaut requires oxygen to breathe
# wrong! this step is missing the connection that humans are a kind of animal. Let's add that.

# Let's check other parts
    sent 1 and int 1: spacesuit backpacks contain oxygen and an astronaut requires oxygen to breathe -> H: An astronaut requires the oxygen in a spacesuit backpack to breathe
# wrong! missing step that there is no oxygen in space. Let's add that"""
    ],
    }

    usage, solutions = task_iterate(wrong_solns, feedbacks)
    for solution in solutions:
        print(solution, '\n')

if __name__ == "__main__":
    test()