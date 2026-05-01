"""LLM client abstraction for data augmentation."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

# -----------------------------------------------------------------------


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    api_key_env: str = "ANTHROPIC_API_KEY"
    prompt_style: str = "quality_enrich"
    batch_size: int = 10
    retry_delay: float = 3.0
    _raw: dict = None  # for round-trip

    @classmethod
    def from_file(cls, path: str | Path) -> "LLMConfig":
        cfg = yaml.safe_load(Path(path).read_text())
        return cls(**{k: v for k, v in cfg.items() if k != "_raw"})

    def to_file(self, path: str | Path):
        Path(path).write_text(yaml.dump(self._raw or yaml.safe_load(
            yml_path=path
        )))


# -----------------------------------------------------------------------
# Quality enrichment prompt
# -----------------------------------------------------------------------

ENRICH_TEMPLATE = """\
You are a data augmentation assistant. For the document below, produce a \
quality-enriched version by appending the following sections:

1. **Summary**: A concise 1-2 sentence summary capturing the core idea.
2. **Key points**: 3-5 bullet points of the most important facts or concepts.
3. **Related questions**: 3-5 follow-up questions a curious reader might ask.

Format the output like this (use the exact numbering, no markdown):

## Summary
[summary text]

## Key points
- [point 1]
- [point 2]
- [point 3]

## Related questions
1. [question 1]
2. [question 2]
3. [question 3]

Then repeat the original document verbatim at the end.

--- Original document ---
{doc}
"""


ENRICH_SYSTEM = (
    "You are a data augmentation assistant. Produce clean, well-structured "
    "enrichment content. Keep everything factual and concise. "
    "Do NOT include any conversational filler or preambles. "
    "Start directly with section 1."
)
