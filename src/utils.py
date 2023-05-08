from abc import ABCMeta, abstractmethod
import time
import traceback
from typing import Callable, Dict, Union


class Prompt:
    def __init__(
        self,
        question_prefix: str,
        answer_prefix: str,
        intra_example_sep: str,
        inter_example_sep: str,
        engine: str = None,
        temperature: float = None,
    ) -> None:
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.intra_example_sep = intra_example_sep
        self.inter_example_sep = inter_example_sep
        self.engine = engine
        self.temperature = temperature

    def make_query(self, prompt: str, question: str) -> str:
        return (
            f"{prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        )


class Feedback(metaclass=ABCMeta):
    def __init__(
        self,
        **kwargs
    ):
        pass

    @abstractmethod
    def make_query(self, solution: str, **kwargs) -> str:
        pass

    @abstractmethod
    def setup_prompt_from_examples_file(self, examples_path: str, **kwargs) -> str:
        pass

    @abstractmethod
    def __call__(self, solution: str, **kwargs) -> Union[str, Dict[str, str]]:
        pass


class FeedbackFactory:
    """ The factory class for feedback generation. """
    registry = {}

    @classmethod
    def create_feedback(cls, name: str, **kwargs) -> 'Feedback':
        pass

    @classmethod
    def register(cls, name: str) -> Callable:
        pass


def retry_parse_fail_prone_cmd(
    func,
    max_retries: int = 100,
    exceptions=(
        ValueError,
        KeyError,
        IndexError,
    ),
):
    def wrapper(*args, **kwargs):
        retries = max_retries
        while retries:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                stack_trace = traceback.format_exc()

                retries -= 1
                print(
                    f"An error occurred: {e}. {stack_trace}. Left retries: {retries}.")
        return None

    return wrapper


def parse_feedback(feedback):
    feedback = feedback.split("\n\n")
    feedback = [f for f in feedback if f.split(
        '\n')[-1].lower() != '# looks good']
    return "\n\n".join(feedback)
