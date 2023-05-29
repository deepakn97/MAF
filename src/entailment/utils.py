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
