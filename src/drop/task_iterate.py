import asyncio
from typing import Dict, List, Tuple, Any
import json
from tqdm import tqdm
from src.utils import Prompt, acall_gpt, call_gpt, extract_answer_gpt


class DropIterate(Prompt):
    def __init__(
        self,
        engine: str,
        prompt_examples: str,
        temperature: float,
        max_tokens: int = 300,
    ) -> None:
        super().__init__(
            question_prefix="Q:",
            answer_prefix="A:",
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

    def make_query(self, solution: Dict[str, str], feedback: Dict[str, str]) -> str:
        query = f"{self.prompt}{self.intra_example_sep}"
        query += f"{self.question_prefix}\n"
        query += "Passage: " + solution["Passage"] + "\n"
        query += "Question: " + solution["Question"]
        query += f"{self.intra_example_sep}{self.answer_prefix}\n"
        query += f"{solution['soln']}{self.intra_example_sep}"

        for feedback_type, feedback_text in feedback.items():
            query += f"{feedback_type}:\n{feedback_text}{self.intra_example_sep}"
        query += f"{self.instruction}"
        return query

    def __call__(
        self,
        solutions: List[Dict[str, str]],
        feedbacks: Dict[str, List[str]],
        batch_size: int = 10,
        concurrent: bool = True,
    ) -> Tuple[Any, List[str]]:
        generation_queries = []
        with open("tmp/log.txt", "a") as f:
            f.write(f"TASK ITERATE SOLUTIONS: {json.dumps(solutions, indent=4)}\n")
            f.write(f"TASK ITERATE FEEDBACKS: {json.dumps(feedbacks, indent=4)}\n")
        print("LENGTH OF SOLUTIONS:", len(solutions))
        for i in range(len(solutions)):
            feedback = {}
            for ft, fb in feedbacks.items():
                feedback[ft] = fb[i]
            generation_queries.append(
                self.make_query(solution=solutions[i], feedback=feedback)
            )

        if not concurrent:
            batch_size = 1

        responses = []
        for i in tqdm(
            range(0, len(generation_queries), batch_size),
            total=len(generation_queries) // batch_size,
        ):
            if concurrent:
                batch_responses = asyncio.run(
                    acall_gpt(
                        generation_queries[i : i + batch_size],
                        self.engine,
                        self.temperature,
                        self.max_tokens,
                        stop_token=self.inter_example_sep,
                    )
                )
            else:
                batch_responses = call_gpt(
                    generation_queries[i : i + batch_size],
                    self.engine,
                    self.temperature,
                    self.max_tokens,
                    stop_token=self.inter_example_sep,
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
    task_iterate = DropIterate(
        engine="text-davinci-003",
        prompt_examples="prompt/drop_maf/iterate.txt",
        temperature=0.0,
    )

    wrong_solns = []
    feedbacks = {
        "Missing Step Feedback": [],
    }

    usage, solutions = task_iterate(wrong_solns, feedbacks)


if __name__ == "__main__":
    test()
