import asyncio
import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any

from tqdm import tqdm
from src.utils import Prompt, acall_gpt, call_gpt


class EntailmentInit(Prompt):
    def __init__(
        self,
        prompt_examples: str,
        engine: str,
        temperature: float,
        max_tokens: int = 300,
    ) -> None:
        super().__init__(
            question_prefix="Hypothesis:",
            answer_prefix="Entailment Tree:\n",
            intra_example_sep="\n\n",
            inter_example_sep="### END ###",
            engine=engine,
            temperature=temperature,
        )
        self.max_tokens = max_tokens
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, prompt_examples) -> str:
        with open(prompt_examples, "r") as f:
            self.prompt = f.read()

    def make_query(self, data: Dict[str, str]) -> str:
        query = f"{self.prompt}\n{self.question_prefix}: {data['hypothesis']}{self.intra_example_sep}"
        for i, sent in enumerate(data["text"]):
            query += f"# sent {i+1}: {sent}\n"
        query = f"{query}\n{self.answer_prefix}"
        return query

    def __call__(
        self, data: List[Dict[str, str]], batch_size=10, concurrent=True
    ) -> Tuple[Any, List[str]]:
        generation_queries = [self.make_query(d) for d in data]
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

        entire_outputs = []
        usage = 0
        finish_reason_stop = 0
        for response in responses:
            if "gpt" in self.engine:
                entire_outputs.append(
                    response["choices"][0]["message"]["content"].strip()
                )
                usage += response["usage"]["total_tokens"]
                finish_reason_stop += response["choices"][0]["finish_reason"] == "stop"
            elif "text-davinci" in self.engine:
                entire_outputs.append(response["choices"][0]["text"].strip())
                usage += response["usage"]["total_tokens"]
                finish_reason_stop += response["choices"][0]["finish_reason"] == "stop"
        print(
            f"Number of times the model finished because of stop token: {finish_reason_stop}/{len(generation_queries)}"
        )

        solutions = []
        for entire_output in entire_outputs:
            if self.inter_example_sep in entire_output:
                solution = entire_output.split(self.inter_example_sep)[0].strip()
                solutions.append(solution)
            else:
                solutions.append(entire_output)

        return usage, solutions


def test():
    task_init = EntailmentInit(
        prompt_examples="prompt/entailment_maf/init.txt",
        engine="text-davinci-003",
        temperature=0.0,
    )

    data = [
        {
            "hypothesis": "Evaporating and condensing can both be caused by changes in heat energy",
            "text": [
                "temperature is a measure of heat energy",
                "temperature changes can cause phase changes",
                "evaporating is a kind of phase change",
                "condensing is a kind of phase change",
            ],
        },
        {
            "hypothesis": "An astronaut requires the oxygen in a spacesuit backpack to breathe",
            "text": [
                "spacesuit backpacks contain oxygen",
                "an animal requires oxygen to breathe",
                "a vacuum does not contain oxygen",
                "an astronaut is a kind of human",
                "a human is a kind of animal",
                "space is a vacuum",
            ],
        },
    ]

    usage, solutions = task_init(data, batch_size=2, concurrent=True)
    for solution in solutions:
        print(solution)
        print("\n")


if __name__ == "__main__":
    test()
