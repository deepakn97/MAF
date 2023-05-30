import argparse
from src.baselines.baseline_utils import (
    BaselineWrapper,
    OPENAI_MODELS,
    TASKS,
    PROMPTS,
    TEMPLATES,
    load_jsonl,
)
import os
from dotenv import load_dotenv
from src.utils import OSModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="text-davinci-003",
        help="Model name to use (e.g. gpt-3.5-turbo, text-davinci-003, vicuna, etc.)",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="gsm_baseline",
        help="Task to run (e.g. gsm_baseline, entailment_baseline, etc.)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="pot_gsm",
        help="Prompt technique to use for the model",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=None,
        help="Path to data file or directory (will go through all jsonl files under directory) otherwise will default based on baseline task",
    )
    parser.add_argument(
        "-n",
        "--num_problems",
        type=int,
        default=None,
        help="number of problems to run (default is all)",
    )
    parser.add_argument(
        "-f", "--format", type=str, default="qa", help="prompt example format"
    )
    parser.add_argument(
        "-c",
        "--cuda_visible",
        type=str,
        default="0, 1, 2",
        help="CUDA_VISIBLE_DEVICES",
    )
    return parser.parse_args()


def main():
    if args.format not in TEMPLATES.keys():
        raise ValueError(f"Format {args.format} not supported")
    baseline = BaselineWrapper(
        args.model,
        args.task,
        args.prompt,
        **TEMPLATES[args.format],
        cuda_visible_devices=args.cuda_visible,
    )
    if args.data is None:
        baseline.run()
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file/directory {args.data} does not exist")
    elif os.path.isdir(args.data):
        for file in os.listdir(args.data):
            if file.endswith(".jsonl"):
                data = load_jsonl(os.path.join(args.data, file))
                baseline.run(
                    data=data,
                    save_file=os.path.join(
                        args.data, file.replace(".jsonl", "_results.json")
                    ),
                )
    else:
        data = load_jsonl(args.data)
        baseline.run(
            data=data,
            save_file=args.data.replace(".jsonl", "_results.json"),
        )


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    main()
