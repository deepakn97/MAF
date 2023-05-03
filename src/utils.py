from time import time
import traceback

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

class LLMFeedback(Prompt):
    def __init__(
        self, 
        question_prefix: str,
        answer_prefix: str,
        intra_example_sep: str,
        inter_example_sep: str,
        engine: str = None,
        temperature: float = None,
        max_tokens: int = 300,
    ) -> None:
        super().__init__(
            question_prefix=question_prefix,
            answer_prefix=answer_prefix,
            intra_example_sep=intra_example_sep,
            inter_example_sep=inter_example_sep,
            engine=engine,
            temperature=temperature,
        )
        self.instruction = """There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."""
        self.max_tokens = max_tokens

def retry_parse_fail_prone_cmd(
    func,
    max_retries: int = 3,
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
                print(f"An error occurred: {e}. {stack_trace}. Left retries: {retries}.")
                time.sleep(60)
        return None

    return wrapper
