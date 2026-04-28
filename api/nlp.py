"""api/nlp.py — Natural Language Processor for opensense.

Input pipeline (applied to every user message before it reaches the model):
  1. Expand informal / chat-speak abbreviations  (u → you, r → are, …)
  2. Normalise punctuation  (!!!→!  ???→?  ALL CAPS words → sentence case)
  3. Spell-correct via pyspellchecker  (skips proper-nouns, short tokens,
     tokens containing digits / special chars, and known technical terms)
  4. Detect user style  (formality, complexity, tone)
  5. Produce a concise style instruction that is injected into the prompt so
     the model mirrors the user's register in its reply.

Output pipeline (applied to the model's reply before it reaches the caller):
  • Strip filler openers the model sometimes prepends  ("Certainly!", …)
  • For casual users, soften overly stiff closing phrases.
  • Deeper adaptation is handled at prompt-injection time via the style
    instruction — the model does the heavy lifting.

Configuration (all flags are controlled via config.yaml → nlp section):
  enabled           — master switch; set false to bypass everything
  spell_check       — run pyspellchecker on input (requires the package)
  style_adapt       — inject style instruction + post-process output
  input_normalize   — expand contractions / fix punctuation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Chat-speak / informal abbreviation expansion map
# ---------------------------------------------------------------------------

_INFORMAL_MAP: list[tuple[str, str]] = [
    (r"\bu\b", "you"),
    (r"\bur\b", "your"),
    (r"\br\b", "are"),
    (r"\bim\b", "I am"),
    (r"\bidk\b", "I don't know"),
    (r"\bimo\b", "in my opinion"),
    (r"\bifaik\b", "as far as I know"),
    (r"\bafaik\b", "as far as I know"),
    (r"\bbtw\b", "by the way"),
    (r"\bfyi\b", "for your information"),
    (r"\bomg\b", "oh my"),
    (r"\blol\b", "haha"),
    (r"\bngl\b", "not going to lie"),
    (r"\bnvm\b", "never mind"),
    (r"\basap\b", "as soon as possible"),
    (r"\bwrt\b", "with regard to"),
    (r"\bw/o\b", "without"),
    (r"\bw/\b", "with"),
    (r"\bcuz\b", "because"),
    (r"\bcause\b", "because"),
    (r"\bgonna\b", "going to"),
    (r"\bwanna\b", "want to"),
    (r"\bgotta\b", "got to"),
    (r"\bkinda\b", "kind of"),
    (r"\bsorta\b", "sort of"),
    (r"\bdunno\b", "don't know"),
    (r"\bplz\b", "please"),
    (r"\bpls\b", "please"),
    (r"\bthx\b", "thank you"),
    (r"\bty\b", "thank you"),
]

# Pre-compile for performance.
_INFORMAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), repl) for pat, repl in _INFORMAL_MAP
]

# ---------------------------------------------------------------------------
# Style-detection marker patterns
# ---------------------------------------------------------------------------

_FORMAL_MARKERS = re.compile(
    r"\b(therefore|furthermore|consequently|hereby|pursuant|notwithstanding|"
    r"regarding|concerning|aforementioned|henceforth|whereby|herein|therein|"
    r"subsequently|respectively|aforementioned)\b",
    re.IGNORECASE,
)

_CASUAL_MARKERS = re.compile(
    r"\b(hey|hi|yo|yeah|yep|nope|cool|awesome|great|sure|ok|okay|"
    r"lol|haha|omg|btw|fyi|gonna|wanna|gotta|kinda|dunno|stuff|thing|"
    r"guys|dude|man|bro|sis)\b",
    re.IGNORECASE,
)

_TECHNICAL_MARKERS = re.compile(
    r"\b(api|http|https|json|xml|sql|nosql|database|algorithm|function|"
    r"variable|parameter|class|module|import|async|await|docker|kubernetes|"
    r"deploy|server|client|backend|frontend|endpoint|token|schema|cache|"
    r"runtime|compiler|parser|regex|array|object|pointer|memory|thread|"
    r"socket|protocol|latency|throughput|payload|serialise|deserialise)\b",
    re.IGNORECASE,
)

# Technical terms that the spell-checker should never alter.
_TECH_TERMS: frozenset[str] = frozenset(
    {
        "api", "apis", "http", "https", "json", "xml", "sql", "html", "css",
        "js", "ts", "jsx", "tsx", "url", "uri", "uuid", "jwt", "oauth",
        "llm", "gpt", "nlp", "ai", "ml", "gpu", "cpu", "ram", "ssd",
        "async", "await", "const", "let", "var", "fn", "str", "int",
        "bool", "dict", "list", "tuple", "enum", "init", "def", "cls",
        "ollama", "llama", "mistral", "phi", "chromadb", "pydantic",
        "fastapi", "uvicorn", "httpx", "yaml", "toml", "env",
    }
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StyleProfile:
    """Detected style of a single user message."""

    formality: Literal["casual", "neutral", "formal"] = "neutral"
    complexity: Literal["simple", "moderate", "technical"] = "moderate"
    tone: Literal["friendly", "neutral", "professional"] = "neutral"
    is_question: bool = False
    corrected_text: str = ""
    original_text: str = ""
    changes_made: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# NLP Processor
# ---------------------------------------------------------------------------


class NLPProcessor:
    """
    Lightweight, zero-heavy-dependency NLP processor.

    The only *optional* third-party package is ``pyspellchecker`` (pure
    Python).  If it is not installed, spell-checking degrades gracefully.
    All other processing is regex / heuristic-based and has no extra deps.
    """

    def __init__(
        self,
        spell_check: bool = True,
        style_adapt: bool = True,
        input_normalize: bool = True,
    ) -> None:
        self.spell_check_enabled = spell_check
        self.style_adapt_enabled = style_adapt
        self.input_normalize_enabled = input_normalize
        self._spell: object | None = None

        if spell_check:
            try:
                from spellchecker import SpellChecker  # type: ignore

                self._spell = SpellChecker()
            except ImportError:
                # Degrade gracefully — spell-check simply won't run.
                self.spell_check_enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_input(self, text: str) -> StyleProfile:
        """
        Normalise, spell-check, and profile the user's raw input.

        The ``corrected_text`` field of the returned :class:`StyleProfile`
        is the AI-friendly version that should be forwarded to the model.
        The ``original_text`` field preserves the raw user message.
        """
        profile = StyleProfile(original_text=text, corrected_text=text)

        working = text

        if self.input_normalize_enabled:
            working, norm_changes = self._normalize(working)
            profile.changes_made.extend(norm_changes)

        if self.spell_check_enabled and self._spell is not None:
            working, spell_changes = self._correct_spelling(working)
            profile.changes_made.extend(spell_changes)

        profile.corrected_text = working

        # Detect whether the message is a question (check corrected text).
        profile.is_question = self._is_question(working)

        # Style detection always runs against the *original* text so that
        # informal markers (expanded away during normalisation) are still
        # picked up for style classification.
        self._detect_style(profile)

        return profile

    def build_style_instruction(self, profile: StyleProfile) -> str:
        """
        Generate a short plain-English instruction for the model so it
        mirrors the user's communication style in its reply.

        Returns an empty string when style adaptation is disabled.
        """
        if not self.style_adapt_enabled:
            return ""

        parts: list[str] = []

        # Tone / formality guidance
        if profile.formality == "casual":
            parts.append(
                "Reply in a friendly, conversational tone — "
                "avoid stiff or overly formal language."
            )
        elif profile.formality == "formal":
            parts.append(
                "Reply in a formal, professional manner with precise language."
            )
        else:
            parts.append("Reply in a clear, helpful, and balanced tone.")

        # Complexity guidance
        if profile.complexity == "simple":
            parts.append(
                "Use plain, everyday language; avoid technical jargon."
            )
        elif profile.complexity == "technical":
            parts.append(
                "The user is comfortable with technical terminology — "
                "you may use it freely."
            )

        # Question handling
        if profile.is_question:
            parts.append(
                "Answer the question directly first, then provide "
                "supporting detail or context."
            )

        return "  ".join(parts)

    def adapt_output(self, text: str, profile: StyleProfile) -> str:
        """
        Light post-processing of the model's raw output.

        Deeper stylistic adaptation is already handled at prompt-injection
        time (via ``build_style_instruction``).  This method only removes
        patterns that models commonly produce regardless of instruction:
        redundant filler openers and overly stiff closing phrases.
        """
        if not self.style_adapt_enabled:
            return text

        # Remove common filler openers that add no value.
        text = re.sub(
            r"^(Certainly!|Of course!|Sure!|Sure thing!|Absolutely!|"
            r"Great question!|That's a great question!|"
            r"I'd be (happy|glad) to help[.!]*|"
            r"I'm here to help[.!]*)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # For casual users, soften stiff formal closers.
        if profile.formality == "casual":
            text = re.sub(
                r"\b(I hope this (adequately )?answers your (query|question)|"
                r"Please do not hesitate to (contact|reach) (me|us)|"
                r"Should you require further (assistance|clarification), "
                r"please (feel free to )?(ask|contact))\b",
                "Hope that helps",
                text,
                flags=re.IGNORECASE,
            )

        return text.strip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> tuple[str, list[str]]:
        """Expand chat-speak and normalise punctuation / casing."""
        changes: list[str] = []

        # 1. Expand informal abbreviations / chat-speak.
        for pattern, replacement in _INFORMAL_PATTERNS:
            new = pattern.sub(replacement, text)
            if new != text:
                changes.append(f"informal expansion")
                text = new

        # 2. Convert excessively capitalised words (≥ 5 all-caps chars)
        #    to capitalised form — likely shouting, not an acronym.
        def _fix_caps(m: re.Match) -> str:
            word = m.group(0)
            # Short all-caps tokens are probably acronyms — leave them.
            if len(word) <= 4:
                return word
            return word.capitalize()

        new = re.sub(r"\b[A-Z]{5,}\b", _fix_caps, text)
        if new != text:
            changes.append("normalised ALL-CAPS words")
            text = new

        # 3. Collapse repeated punctuation:  !!!→!  ???→?  ...→…
        new = re.sub(r"([!?])\1{2,}", r"\1", text)
        if new != text:
            changes.append("normalised repeated punctuation")
            text = new

        # 4. Strip excessive whitespace.
        text = re.sub(r" {2,}", " ", text).strip()

        return text, changes

    def _correct_spelling(self, text: str) -> tuple[str, list[str]]:
        """
        Word-by-word spell correction using pyspellchecker.

        Tokens are skipped when they:
        - Contain digits, punctuation, or non-ASCII characters.
        - Are 2 characters or shorter (likely abbreviations or initials).
        - Start with a capital letter (likely proper nouns).
        - Appear in the known technical-terms allow-list.
        """
        from spellchecker import SpellChecker  # type: ignore

        spell: SpellChecker = self._spell  # type: ignore
        changes: list[str] = []

        # Split on whitespace while preserving delimiters for reconstruction.
        tokens = re.split(r"(\s+)", text)
        result: list[str] = []

        for token in tokens:
            # Only attempt correction on lowercase word tokens (≥ 3 chars,
            # pure ASCII alpha, no apostrophe-only tokens).
            if not re.match(r"^[a-z]{3,}$", token):
                result.append(token)
                continue

            if token.lower() in _TECH_TERMS:
                result.append(token)
                continue

            if spell.unknown([token]):
                correction = spell.correction(token)
                if correction and correction != token:
                    changes.append(f"spell: '{token}' → '{correction}'")
                    result.append(correction)
                    continue

            result.append(token)

        return "".join(result), changes

    @staticmethod
    def _is_question(text: str) -> bool:
        """Heuristically determine if the message is a question."""
        stripped = text.rstrip()
        if stripped.endswith("?"):
            return True
        # Question words in the first 80 characters.
        if re.search(
            r"\b(what|who|where|when|why|how|is|are|can|could|would|"
            r"should|do|does|did|will|has|have)\b",
            text[:80],
            re.IGNORECASE,
        ):
            return True
        return False

    @staticmethod
    def _detect_style(profile: StyleProfile) -> None:
        """Populate formality / complexity / tone on *profile* in-place."""
        text = profile.original_text

        formal_hits = len(_FORMAL_MARKERS.findall(text))
        casual_hits = len(_CASUAL_MARKERS.findall(text))
        technical_hits = len(_TECHNICAL_MARKERS.findall(text))

        # --- Formality ---
        if formal_hits > casual_hits and formal_hits >= 1:
            profile.formality = "formal"
        elif casual_hits > formal_hits and casual_hits >= 1:
            profile.formality = "casual"
        else:
            profile.formality = "neutral"

        # --- Complexity ---
        if technical_hits >= 2:
            profile.complexity = "technical"
        else:
            words = [w for w in text.split() if w.isalpha()]
            if words:
                avg_len = sum(len(w) for w in words) / len(words)
                profile.complexity = "simple" if avg_len < 4.5 else "moderate"
            else:
                profile.complexity = "moderate"

        # --- Tone (derived from formality) ---
        if profile.formality == "casual":
            profile.tone = "friendly"
        elif profile.formality == "formal":
            profile.tone = "professional"
        else:
            profile.tone = "neutral"
