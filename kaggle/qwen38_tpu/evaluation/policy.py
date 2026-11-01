"""Explicit runtime budgets and output instructions; reference answers are never read."""

import copy

QUALITY_POLICY = "quality_first_v2"

FINAL_CONTRACT = (
    "Keep all explanations inside the thinking section. In the final answer, follow "
    "the user's requested format exactly, without introductory or concluding prose. "
    "For a requested list of numbers, emit only that list. For JSON or JSONL, emit "
    "only the requested records, without Markdown fences."
)
ANNOTATION_CONTRACT = (
    " A StructuredAnnotation has exactly four top-level keys: subject_id, timestamp_ms, "
    "claims, confidence. The confidence field belongs at the top level, outside claims. "
    "Include only the claims requested by the user."
)
PAIR_CONTRACT = (
    " In the pair review, provide between one and six reason bullets. "
    "Do not use either angle-bracket character in the final answer, including "
    "ASCII arrows; write temporal changes in words."
)


def quality_case(original):
    """Preserve source inputs/labels and record every deliberate protocol change."""
    case = copy.deepcopy(original)
    pipeline = case.get("group") == "beamng_pipeline_review"
    case["max_new_tokens"] = 8192 if pipeline else 4096
    case["answer_reserve_tokens"] = 2048 if pipeline else 1024
    case["system_prompt"] = FINAL_CONTRACT
    if case["match"]["kind"] == "annotation_json":
        case["system_prompt"] += ANNOTATION_CONTRACT
    if case["match"]["kind"] == "pair_markdown":
        case["system_prompt"] += PAIR_CONTRACT
    if case["images"] and case.get("group") in {"text_reading", "sft_formatting"}:
        case["min_pixels"] = 1048576
        case["max_pixels"] = max(1048576, case.get("max_pixels", 589824))
    case["runtime_policy"] = dict(
        name=QUALITY_POLICY,
        original_max_new_tokens=original.get("max_new_tokens"),
        max_new_tokens=case["max_new_tokens"],
        final_answer_reserve_tokens=case["answer_reserve_tokens"],
        original_min_pixels=original.get("min_pixels", 65536),
        original_max_pixels=original.get("max_pixels", 589824),
        min_pixels=case.get("min_pixels", 65536),
        max_pixels=case.get("max_pixels", 589824),
        system_contract_added=True,
        source_prompt_unchanged=True,
        source_images_unchanged=True,
        matcher_unchanged=True,
        limitation="Explicit output contract v2; compare separately with v11/v12 protocols.",
    )
    return case


class AnswerReserve:
    """Close reasoning at its budget, leaving real model steps for the final answer.

    The inserted closing delimiter passes through normal ngram lookup and model
    state updates. No answer content is inserted, repaired or scored differently.
    """

    def __init__(self, tokenizer, prompts, limits, eos):
        self.reserves = [p["case"].get("answer_reserve_tokens", 0) for p in prompts]
        self.budgets = []
        for reserve, limit in zip(self.reserves, limits, strict=True):
            if type(reserve) is not int or reserve < 0 or (reserve and reserve > int(limit) - 2):
                raise ValueError("Final-answer reserve must leave room for reasoning and delimiter")
            self.budgets.append(int(limit) - reserve - 1)
        tokens = tokenizer.encode("</think>", add_special_tokens=False)
        if len(tokens) != 1 or tokens[0] in eos:
            raise ValueError("Thinking budget requires a single non-EOS closing delimiter")
        if tokenizer.decode(tokens, skip_special_tokens=False) != "</think>":
            raise ValueError("Tokenizer does not round-trip the thinking closing delimiter")
        self.end = tokens[0]
        self.ended = [False] * len(prompts)
        self.forced = [False] * len(prompts)

    def select(self, ids, active, output):
        ids = ids.copy()
        for row in range(len(ids)):
            if not active[row] or self.ended[row]:
                continue
            if self.reserves[row] and len(output[row]) >= self.budgets[row]:
                ids[row] = self.end
                self.forced[row] = True
            if ids[row] == self.end:
                self.ended[row] = True
        return ids
