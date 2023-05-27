from typing import List

def get_entailment_proof(generated_trees: List[str]) -> List[str]:
    proofs = []
    for tree in generated_trees:
        proof_steps = []
        tree = tree.strip()
        steps = tree.split("\n")
        for step in steps:
            if "->" in step:
                step_split = step.split("->")
                lhs = step_split[0].split(':')[0].strip()
                lhs.replace("and", "&")
                rhs = step_split[1].strip()
                if "hypothesis" in rhs:
                    rhs = rhs.split(":")[0].strip()
                proof_steps.append(f"{lhs} -> {rhs}")
            else:
                continue
        proofs.append(("; ".join(proof_steps)).strip())
    return proofs
