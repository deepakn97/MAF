import os
from typing import Dict, List
from src.utils import LLMFeedback, OSFeedback, FeedbackFactory, Feedback


class EntailmentLLMFeedback(LLMFeedback):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            question_prefix="Hypothesis: ",
            answer_prefix="Entailment Tree:",
            intra_example_sep="\n\n",
            inter_example_sep="### END ###",
            **kwargs,
        )

    def make_query(self, data: Dict[str, str]) -> str:
        query = f"{self.prompt}\n{self.question_prefix}{data['hypothesis']}{self.intra_example_sep}"
        for i, sent in enumerate(data["text"]):
            query += f"# sent {i+1}: {sent}\n"
        query = f"{query}\n{self.answer_prefix}\n{data['soln']}{self.instruction}{self.instruction}"
        return query

    def process_outputs(self, outputs: List[str]) -> List[str]:
        fb_and_solns = []
        for entire_output in outputs:
            if "### END" in entire_output:
                entire_output = entire_output.split("### END")[0]
            fb_and_maybe_soln = entire_output.strip()
            if self.eager_refine:
                if self.answer_prefix in fb_and_maybe_soln:
                    feedback = fb_and_maybe_soln.split(self.answer_prefix)[0].strip()
                    solution = fb_and_maybe_soln.split(self.answer_prefix)[1].strip()
                else:
                    feedback = fb_and_maybe_soln
                    solution = ""
            else:
                feedback = fb_and_maybe_soln
                solution = ""
            fb_and_solns.append({"feedback": feedback, "solution": solution})

        return fb_and_solns


@FeedbackFactory.register("self_refine")
class SelfRefineFeedback(EntailmentLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(name="", max_tokens=600, eager_refine=True, **kwargs)
        self.instruction = "There is an error in the entailment tree above because of lack of understanding of the context. What is the error? To find the error, go through each step of the entailment tree, and check if everything looks good."
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("missing_step")
class MissingStepFeedback(EntailmentLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(
            name="Missing Step Feedback", max_tokens=600, eager_refine=True, **kwargs
        )
        self.instruction = "# Check the above entailment tree line by line for any missing steps and suggest fixes. Ignore all other types of errors."
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("commonsense")
class CommonsenseFeedback(EntailmentLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(
            name="Commonsense Feedback", max_tokens=600, eager_refine=True, **kwargs
        )
        self.instruction = ""
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("repetition")
class RepetitionFeedback(EntailmentLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(
            name="Repetition Feedback", max_tokens=600, eager_refine=True, **kwargs
        )
        self.instruction = "# Check the above entailment tree line by line for any repetition errors and suggest fixes. Ignore all other types of errors."
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("redundancy")
class RedundancyFeedback(EntailmentLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(
            name="Redundancy Feedback", max_tokens=600, eager_refine=True, **kwargs
        )
        self.instruction = "# Check the above entailment tree line by line for any redundant steps and suggest fixes. Ignore all other types of errors."
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("irrelevancy")
class IrrelevancyFeedback(EntailmentLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(
            name="Irrelevancy Feedback", max_tokens=600, eager_refine=True, **kwargs
        )
        self.instruction = "# Check the above entailment tree line by line for any irrelevant information and suggest fixes. Ignore all other types of errors."
        self.setup_prompt_from_examples_file(prompt_examples)


def test():
    wrong_solns = [
        {
            "hypothesis": "Evaporating and condensing can both be caused by changes in heat energy",
            "text": [
                "temperature is a measure of heat energy",
                "temperature changes can cause phase changes",
                "evaporating is a kind of phase change",
                "condensing is a kind of phase change",
            ],
            "soln": """sent 1 and sent 2: temperature is a measure of heat energy and temperature changes can cause phase changes -> int 1: changes in heat energy can cause phase changes
int 1 and sent 3 and sent 4: changes in heat energy can cause phase changes and evaporating is a kind of phase change and condensing is a kind of phase change -> H: evaporating and condensing can both be caused by changes in heat energy""",
        },
        {
            "hypothesis": "An astronaut requires the oxygen in a spacesuit backpack to breathe",
            "text": [
                "spacesuit backpacks contain oxygen",
                "an animal requires oxygen to breathe",
                "a vacuum does not contain oxygen",
                "an astronaut is a kind of human",
                "a human is a kind of animal",
                "space is a vacuum" "space is cold",
            ],
            "soln": """sent 4 and sent 5: an astronaut is a kind of human and a human is a kind of animal -> int 1: an astronaut is a kind of animal
sent 3 and sent 6: a vacuum does not contain oxygen and space is a vacuum -> int 2: there is no oxygen in space
sent 2 and int 1: an animal requires oxygen to breathe and an astronaut is a kind of animal -> int 3: an astronaut requires oxygen to breathe
sent 6 and sent 7: space is a vaccum and space is cold -> int 4: space is a cold vaccum
sent 1 and int 2 and int 3: spacesuit backpacks contain oxygen and there is no oxygen in space and an astronaut requires oxygen to breathe -> H: an astronaut requires the oxygen in spacesuit backpack to breathe""",
        },
    ]
    # ----- OpenAI Models ----- #
    openai_feedbacks = [
        ft
        for ft in list(FeedbackFactory.registry.keys())
        if "os" not in ft and ft != "self_refine"
    ]
    for feedback in openai_feedbacks:
        fb_prompt_path = os.path.join("prompt/entailment_maf/", f"{feedback}.txt")
        fm = FeedbackFactory.create_feedback(
            feedback,
            prompt_examples=fb_prompt_path,
            engine="text-davinci-003",
            temperature=0.0,
        )
        usage, fb_and_maybe_solns = fm(wrong_solns)
        for i, fb_and_soln in enumerate(fb_and_maybe_solns):
            print(f"{fm.name} Feedback {i}:\n{fb_and_soln['feedback']}\n")
            print(f"{fm.name} Solution {i}:\n{fb_and_soln['solution']}\n")

    # ----- OS Models ----- #
    os_feedbaks = [
        ft
        for ft in list(FeedbackFactory.registry.keys())
        if "os" in ft and ft != "self_refine"
    ]


if __name__ == "__main__":
    test()
