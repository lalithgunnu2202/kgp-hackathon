from src.llm_judge import llm_verify_claim

HIGH_THRESHOLD = 0.45
LOW_THRESHOLD = 0.30


def decide_claim(score, claim=None, evidence_chunks=None):
    if score >= HIGH_THRESHOLD:
        return True
    elif score <= LOW_THRESHOLD:
        return False
    else:
        if claim is None or evidence_chunks is None:
            return False
        return llm_verify_claim(claim, evidence_chunks)


def final_decision(claim_results, support_ratio=0.6):
    return sum(claim_results) / len(claim_results) >= support_ratio


