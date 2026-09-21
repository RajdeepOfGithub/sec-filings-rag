"""
Citation verification: for each [n] in an answer, ask whether the cited passage
actually supports the claim attached to that citation number.

This checks support, not existence. A citation can resolve to a real chunk and
still fail here, because the chunk does not say what the answer claims it says.

One gpt-4o-mini call per citation occurrence, temperature 0 (call_llm hardcodes
it). Cost scales with the number of citation markers in the answer.
"""
from generator import CITATION_PATTERN, AnswerResult, call_llm

VERIFY_MODEL = "gpt-4o-mini"

VERIFY_SYSTEM_PROMPT = """You check whether a source passage supports a claim.

The passage is from an SEC filing and may contain flattened tables. Supported means every figure, period and entity in the claim is stated in the passage or reads directly off it. A passage on the right topic that lacks the claimed figure, or that states it for a different period, does not support the claim.

Answer on one line, exactly: YES - <reason> or NO - <reason>. Keep the reason under 20 words."""


def _parse_verdict(reply):
    """Anything that does not start with YES counts as unsupported."""
    text = (reply or "").strip()
    head, _, reason = text.partition("-")
    supported = head.strip().upper().startswith("YES")
    return {"supported": supported, "reason": reason.strip() or text}

def verify_citation(claim_text, cited_chunk_text):
    """Returns {"supported": bool, "reason": str} for one (claim, passage) pair."""
    user_prompt = (
        f"Passage:\n{cited_chunk_text}\n\n"
        f"Claim:\n{claim_text}\n\n"
        "Does the passage support the claim?"
    )
    return _parse_verdict(call_llm(VERIFY_SYSTEM_PROMPT, user_prompt, model=VERIFY_MODEL))

def split_claims(answer_text):
    """
    Splits an answer into (claim, number) pairs, one per citation occurrence.

    The claim for a marker is the text between the previous marker and this one:
    "Net income was X [2]." makes "Net income was X" the claim for 2. Adjacent
    markers ([4][5], or [1, 2]) share the claim that precedes them, so both get
    checked against the same text.
    """
    pairs = []
    previous_end = 0
    previous_claim = ""

    for match in CITATION_PATTERN.finditer(answer_text):
        claim = answer_text[previous_end:match.start()].strip()
        if not claim:
            claim = previous_claim
        for part in match.group(1).split(","):
            pairs.append({"number": int(part), "claim": claim})
        previous_end = match.end()
        previous_claim = claim

    return pairs

def verify_answer(answer_result: AnswerResult) -> AnswerResult:
    """
    Verifies every citation occurrence in answer_result.answer_text and attaches
    the results as answer_result.citation_verification. Returns the same object.
    """
    chunk_id_by_number = {c["number"]: c["chunk_id"] for c in answer_result.citations}
    text_by_chunk_id = {r["id"]: r["text"] for r in answer_result.retrieved_chunks}

    verifications = []
    for pair in split_claims(answer_result.answer_text):
        chunk_id = chunk_id_by_number.get(pair["number"])
        chunk_text = text_by_chunk_id.get(chunk_id)

        if chunk_text is None:
            verdict = {"supported": False, "reason": "citation number does not resolve to a context passage"}
        else:
            verdict = verify_citation(pair["claim"], chunk_text)

        verifications.append({
            "number": pair["number"],
            "chunk_id": chunk_id,
            "claim": pair["claim"],
            "supported": verdict["supported"],
            "reason": verdict["reason"],
        })

    answer_result.citation_verification = verifications
    return answer_result
