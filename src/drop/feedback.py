import os
from typing import Dict, List
from src.utils import LLMFeedback, OSFeedback, FeedbackFactory, Feedback


class DropLLMFeedback(LLMFeedback):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            question_prefix="Q:",
            answer_prefix="A:",
            intra_example_sep="\n\n",
            inter_example_sep="### END ###",
            **kwargs,
        )

    def make_query(self, data: Dict[str, str]) -> str:
        query = f"{self.prompt}{self.intra_example_sep}"
        query += f"{self.question_prefix}\n"
        query += "Passage: " + data["Passage"] + "\n"
        query += "Question: " + data["Question"]
        query += f"{self.intra_example_sep}{self.answer_prefix}\n"
        query += f"{data['soln']}\n\n{self.instruction}"
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
class SelfRefineFeedback(DropLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(name="", max_tokens=600, eager_refine=True, **kwargs)
        self.instruction = "There is an error in the answer above because of lack of understanding of the problem. What is the error? To find the error, go through the reasoning and check if everything looks good."
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("missing_step")
class MissingStepFeedback(DropLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(
            name="Missing Step Feedback", max_tokens=600, eager_refine=True, **kwargs
        )
        self.instruction = "# There is an error in the answer above because of missing steps. What is the error? To find the error, go through the reasoning and check if everything looks good. Ignore all other types of errors."
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("commonsense")
class CommonsenseFeedback(DropLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(
            name="Commonsense Feedback", max_tokens=600, eager_refine=True, **kwargs
        )
        self.instruction = "# There is a commonsense reasoning error in the answer above. What is the error? To find the error, go through the reasoning and check if everything looks good. Ignore all other types of errors."
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("repetition")
class RepetitionFeedback(DropLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(
            name="Repetition Feedback", max_tokens=600, eager_refine=True, **kwargs
        )
        self.instruction = "# There is an error in the answer above because of repetitions. What is the error? To find the error, go through the reasoning and check if everything looks good. Ignore all other types of errors."
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("redundancy")
class RedundancyFeedback(DropLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(
            name="Redundancy Feedback", max_tokens=600, eager_refine=True, **kwargs
        )
        self.instruction = "# There is an error in the answer above because of redundant steps. What is the error? To find the error, go through the reasoning and check if everything looks good. Ignore all other types of errors."
        self.setup_prompt_from_examples_file(prompt_examples)


@FeedbackFactory.register("irrelevancy")
class IrrelevancyFeedback(DropLLMFeedback):
    def __init__(self, prompt_examples: str, **kwargs) -> None:
        super().__init__(
            name="Irrelevancy Feedback", max_tokens=600, eager_refine=True, **kwargs
        )
        self.instruction = "# There is an error in the answer above because of irrelevant information. What is the error? To find the error, go through the reasoning and check if everything looks good. Ignore all other types of errors."
        self.setup_prompt_from_examples_file(prompt_examples)


def test():
    wrong_solns = []
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
