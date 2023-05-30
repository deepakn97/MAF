from utils.eval_utils import score_prediction_whole_proof
from eval.run_scorer import shortform_angle
from utils.angle_utils import decompose_slots
from bleurt import score
import json
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="file to evaluate (must be JSON file with a list of objects, each with the keys 'id', 'angle', 'prediction', and 'gold' where 'gold' contains the entire gold entailment dictionary (found in processed_data/slots or public_dataset))",
    )
    parser.add_argument(
        "-o", "--outfile", type=str, required=True, help="file to write results to"
    )
    parser.add_argument(
        "-b",
        "--bleurt_checkpoint",
        type=str,
        default="/data4/d_wang/nlp/models/bleurt-large-512",
        help="path to bleurt checkpoint",
    )
    return parser.parse_args()


def individual_eval(id, prediction, angle, gold, bleurt_scorer=None):
    pred_dict = {
        "id": id,
        "angle": angle,
        "prediction": prediction,
        "angle_str": shortform_angle(angle, sort_angle=False),
        "slots": decompose_slots(prediction),
    }
    prediction_json = ""
    scoring_spec = {
        "hypothesis_eval": "nlg",
        "proof_eval": "entail_whole_proof_align_eval",
        # "proof_eval": "entail_whole_polish_proof_align_eval",
    }
    metrics = score_prediction_whole_proof(
        pred_dict,
        gold,
        prediction_json=prediction_json,
        scoring_spec=scoring_spec,
        bleurt_scorer=bleurt_scorer,
    )
    return metrics


def main():
    with open(args.file) as f:
        data = json.load(f)
    results = []
    for d in data:
        id = d["id"]
        prediction = d["prediction"]
        angle = d["angle"]
        gold = d["gold"]
        metrics = individual_eval(id, prediction, angle, gold, args.bleurt_checkpoint)
        results.append(metrics)
    with open(args.outfile, "w") as f:
        json.dump(results, f, indent=4)


# MAKE SURE TO USE ABSPATH FOR THIS
if __name__ == "__main__":
    args = parse_args()
    main()
