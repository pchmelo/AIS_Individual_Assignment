"""
Pair-selection helpers for intersectional fairness analysis.

Holds the prompt template and response-parsing logic that were previously
inlined in ``pipeline.py._select_best_pairs``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


# ── Prompt template ────────────────────────────────────────────────────

PAIR_SELECTION_PROMPT_TEMPLATE = """\
    You are analyzing a dataset for fairness. The following sensitive attributes were detected:
    {sensitive_cols_str}

    All possible attribute pairs for intersectional analysis ({total_pairs} total):
    {pair_list_str}

    Select exactly {max_pairs} pairs that are MOST IMPORTANT for fairness analysis. Consider:
    1. Historical discrimination patterns (e.g., Race+Sex, Age+Gender are commonly studied)
    2. Potential for intersectional bias (combinations that may compound disadvantage)
    3. Relevance to employment/lending/healthcare fairness (depending on context)
    4. Statistical significance (pairs that likely have enough data points)

    Respond in this EXACT format:
    SELECTED_PAIRS:
    - Attribute1 + Attribute2
    - Attribute3 + Attribute4

    REASONING:
    <Your explanation of why these pairs were selected>
"""


def build_pair_selection_prompt(
    sensitive_cols: List[str],
    all_pairs: List[Tuple[str, str]],
    max_pairs: int,
) -> str:
    """Build the prompt asking the agent to choose the best pairs."""
    pair_list_str = "\n".join([f"  - {p[0]} + {p[1]}" for p in all_pairs])
    return PAIR_SELECTION_PROMPT_TEMPLATE.format(
        sensitive_cols_str=", ".join(sensitive_cols),
        total_pairs=len(all_pairs),
        pair_list_str=pair_list_str,
        max_pairs=max_pairs,
    )


# ── Response parser ────────────────────────────────────────────────────

def parse_pair_selection_response(
    response: str,
    all_pairs: List[Tuple[str, str]],
    max_pairs: int,
) -> Tuple[List[Tuple[str, str]], str]:
    """Parse the agent response and return ``(selected_pairs, reasoning)``.

    Falls back to the first *max_pairs* pairs if parsing does not yield
    enough results.
    """
    selected: List[Tuple[str, str]] = []
    reasoning = ""
    lines = response.strip().split("\n")
    in_pairs = False
    in_reasoning = False
    reasoning_lines: List[str] = []

    for line in lines:
        line_stripped = line.strip()
        if "SELECTED_PAIRS:" in line.upper():
            in_pairs = True
            in_reasoning = False
            continue
        if "REASONING:" in line.upper():
            in_pairs = False
            in_reasoning = True
            continue

        if in_pairs and line_stripped.startswith("-"):
            # Parse "- Attr1 + Attr2" format
            pair_str = line_stripped[1:].strip()
            if "+" in pair_str:
                parts = [p.strip() for p in pair_str.split("+")]
                if len(parts) == 2:
                    # Find matching pair (order-independent, case-insensitive)
                    for p in all_pairs:
                        if (parts[0].lower() == p[0].lower() and parts[1].lower() == p[1].lower()) or \
                           (parts[0].lower() == p[1].lower() and parts[1].lower() == p[0].lower()):
                            if p not in selected:
                                selected.append(p)
                            break

        if in_reasoning:
            if line_stripped and not line_stripped.upper().startswith("REASONING:"):
                reasoning_lines.append(line_stripped)

    reasoning = " ".join(reasoning_lines).strip()

    # Validate we got enough pairs
    if len(selected) == 0:
        selected = list(all_pairs[:max_pairs])
        reasoning = f"Automatic selection: first {max_pairs} pairs (agent parsing issue). Agent reasoning was: {reasoning}"
    elif len(selected) < max_pairs:
        # Agent found some pairs, but fewer than max_pairs. Append missing ones up to max_pairs.
        for p in all_pairs:
            if len(selected) >= max_pairs:
                break
            if p not in selected:
                selected.append(p)
    elif len(selected) > max_pairs:
        selected = selected[:max_pairs]

    return selected, reasoning
