from typing import List


def get_entailment_proof(generated_trees: List[str]) -> List[str]:
    proofs = []
    for tree in generated_trees:
        proof_steps = []
        tree = tree.strip()
        steps = tree.split("\n")
        for step in steps:
            if "->" in step:
                # fix bug where llm will output sentence labels as 'sent 1' instead of 'sent1'
                full_split = step.split(" ")
                full = []
                i = 0
                while i < len(full_split):
                    if (full_split[i] in ["sent", "int"]) and i + 1 < len(full_split):
                        full_split[i] += full_split[i + 1]
                        full.append(full_split[i])
                        i += 1
                    else:
                        full.append(full_split[i])
                    i += 1
                step = " ".join(full)
                step_split = step.split("->")
                lhs = step_split[0].split(":")[0].strip()
                lhs = lhs.replace("and", "&")
                rhs = step_split[1].strip()
                if "hypothesis" in rhs:
                    rhs = rhs.split(":")[0].strip()

                proof_steps.append(f"{lhs} -> {rhs}")
            else:
                continue
        proofs.append(("; ".join(proof_steps)).strip())
    return proofs


def extract_eager(feedbacks: List[str]) -> List[str]:
    rewrites = []
    for feedback in feedbacks:
        key = "here is the rewrite"
        # find the line with "here is the rewrite" and take all lines up to that until "### END ###"
        if key in feedback.lower():
            feedback = feedback.lower()
            feedback = feedback.split(key)[1]
            feedback = feedback.split("### end ###")[0]
            feedback = feedback.split("\n")
            feedback = [f.strip() for f in feedback[1:]]
            feedback = '\n'.join([f for f in feedback if f != ""])
            rewrites.append(feedback)
        else:
            rewrites.append("")
    return rewrites
