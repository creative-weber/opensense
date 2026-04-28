"""
api/authenticator.py — Authenticity and quality gate for new information.

Before a fact is committed to memory it must pass this gate.  Two layers run
in sequence:

    1. Heuristic checks (fast, no network) — rejects obvious garbage.
    2. Model-assisted check (optional) — asks the running LLM whether the
       text looks like a credible, factual statement.  Can be disabled in
       config so the API still works without a model server running.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import httpx

# ------------------------------------------------------------------
# Public types
# ------------------------------------------------------------------

Verdict = Literal["accepted", "rejected", "uncertain"]


@dataclass
class AuthResult:
    verdict: Verdict
    confidence: float          # 0.0 – 1.0
    reason: str


# ------------------------------------------------------------------
# Heuristic layer
# ------------------------------------------------------------------

_MIN_CHARS = 20
_MAX_CHARS = 4_000
# Ratio of non-alpha characters that triggers a gibberish flag
_GIBBERISH_THRESHOLD = 0.6
# Patterns that strongly suggest prompt injection or adversarial input
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak|evil)", re.I),
    re.compile(r"<\|(?:system|user|assistant)\|>", re.I),
    re.compile(r"\[INST\]|\[/INST\]", re.I),
    re.compile(r"system\s*prompt\s*:", re.I),
]


def _heuristic_check(text: str) -> AuthResult:
    stripped = text.strip()

    if len(stripped) < _MIN_CHARS:
        return AuthResult("rejected", 0.0, "Too short to be a meaningful fact.")

    if len(stripped) > _MAX_CHARS:
        return AuthResult("rejected", 0.0, "Exceeds maximum allowed length.")

    # Gibberish detector: count ratio of non-letter, non-digit, non-space chars
    non_word = sum(1 for c in stripped if not (c.isalpha() or c.isdigit() or c.isspace()))
    ratio = non_word / max(len(stripped), 1)
    if ratio > _GIBBERISH_THRESHOLD:
        return AuthResult("rejected", 0.0, f"High symbol ratio ({ratio:.0%}); likely gibberish.")

    # Prompt injection guard
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            return AuthResult(
                "rejected",
                0.0,
                "Input resembles a prompt-injection attempt and was blocked.",
            )

    # Must contain at least one verb-like word (crude sentence check)
    words = stripped.split()
    if len(words) < 4:
        return AuthResult("rejected", 0.1, "Too few words to constitute a fact.")

    return AuthResult("accepted", 0.75, "Passed heuristic checks.")


# ------------------------------------------------------------------
# Model-assisted layer
# ------------------------------------------------------------------

_VERIFY_PROMPT = """\
You are a fact-validation assistant. Evaluate the following statement and decide \
whether it reads as a plausible, authentic piece of information (it does not have \
to be proven true — it just must not be nonsense, fictional, or adversarial).

Statement: {statement}

Respond with EXACTLY one of:
  AUTHENTIC   — the statement is coherent and plausible
  UNCERTAIN   — the statement is ambiguous or unverifiable
  FAKE        — the statement is clearly nonsensical, fictional, or adversarial

Then on a new line write a one-sentence reason.
Example response:
AUTHENTIC
The statement is a well-formed factual claim about a real-world topic.
"""


async def _model_check(
    text: str,
    backend: str,
    ollama_url: str,
    llamacpp_url: str,
    model_name: str,
) -> AuthResult:
    prompt = _VERIFY_PROMPT.format(statement=text)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if backend == "ollama":
                r = await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
                r.raise_for_status()
                raw = r.json().get("response", "")
            else:
                r = await client.post(
                    f"{llamacpp_url}/completion",
                    json={"prompt": prompt, "n_predict": 80, "stream": False},
                )
                r.raise_for_status()
                raw = r.json().get("content", "")
    except Exception as exc:
        # Model server unavailable — fall back to heuristic result
        return AuthResult("uncertain", 0.5, f"Model check skipped: {exc}")

    upper = raw.strip().upper()
    lines = raw.strip().splitlines()
    reason = lines[1].strip() if len(lines) > 1 else raw.strip()

    if upper.startswith("AUTHENTIC"):
        return AuthResult("accepted", 0.95, reason)
    if upper.startswith("FAKE"):
        return AuthResult("rejected", 0.05, reason)
    return AuthResult("uncertain", 0.5, reason)


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

async def authenticate(
    text: str,
    *,
    use_model: bool = True,
    backend: str = "ollama",
    ollama_url: str = "http://localhost:11434",
    llamacpp_url: str = "http://localhost:8080",
    model_name: str = "my-custom-model",
    min_confidence: float = 0.6,
) -> AuthResult:
    """
    Run the full authenticity pipeline on *text*.

    Pipeline:
        heuristic check  →  (if use_model) model-assisted check
        →  final verdict based on min_confidence threshold

    Parameters
    ----------
    text            : The information to validate.
    use_model       : Whether to run the model-assisted check.
    backend         : "ollama" | "llamacpp"
    ollama_url      : Ollama server base URL.
    llamacpp_url    : llama.cpp server base URL.
    model_name      : Model to use for verification.
    min_confidence  : Minimum confidence required to accept the fact (0–1).
    """
    # Layer 1 — heuristics
    h = _heuristic_check(text)
    if h.verdict == "rejected":
        return h

    if not use_model:
        return h

    # Layer 2 — model
    m = await _model_check(text, backend, ollama_url, llamacpp_url, model_name)

    # Merge: model result wins when available; heuristic acts as floor
    combined_confidence = (h.confidence * 0.3) + (m.confidence * 0.7)

    if m.verdict == "rejected" or combined_confidence < min_confidence:
        return AuthResult(
            "rejected",
            round(combined_confidence, 3),
            m.reason or h.reason,
        )

    if m.verdict == "uncertain":
        return AuthResult("uncertain", round(combined_confidence, 3), m.reason)

    return AuthResult("accepted", round(combined_confidence, 3), m.reason)
