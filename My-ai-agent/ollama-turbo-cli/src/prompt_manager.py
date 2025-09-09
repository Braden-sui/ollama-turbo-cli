"""Centralized prompt management for Ollama Turbo CLI.

Provides consistent, versioned prompts for:
- Initial system message
- Post-tool reprompting
- Mem0 context injection block

Behavior can be adjusted via environment variables without touching core logic.
"""
from __future__ import annotations

import os
from typing import List
from . import plugin_loader


class PromptManager:
    """Builds and returns prompt strings used across the client."""

    def __init__(self, reasoning: str, *, verbosity: str | None = None, verbose_after_tools: bool | None = None, fewshots: bool | None = None) -> None:
        self.reasoning = reasoning
        # Defaults preserve previous behavior (concise, no fewshots, compact after-tools reprompt)
        try:
            self.verbosity = (verbosity or os.getenv("PROMPT_VERBOSITY", "concise") or "concise").lower()
        except Exception:
            self.verbosity = "concise"
        try:
            self.verbose_after_tools = bool(verbose_after_tools if verbose_after_tools is not None else ((os.getenv("PROMPT_VERBOSE_AFTER_TOOLS") or "0").lower() in {"1","true","yes","on"}))
        except Exception:
            self.verbose_after_tools = False
        try:
            self.fewshots = bool(fewshots if fewshots is not None else ((os.getenv("PROMPT_FEWSHOTS") or "0").lower() in {"1","true","yes","on"}))
        except Exception:
            self.fewshots = False

    # ---------- System / Initial Prompt ----------
    def initial_system_prompt(self) -> str:
        verbosity = (self.verbosity or "concise").lower()
        style_line = "Be thorough and structured." if verbosity == "detailed" else "Be concise but complete."
        base = (
            "You are GPT-OSS running with Harmony channels.\n\n"
            "â€” Reasoning â€”\n"
            f"â€¢ Reasoning: {self.reasoning}\n\n"
            "â€” Safety/Process â€”\n"
            "â€¢ Keep internal reasoning private (no chain-of-thought). Provide only final answers and short, audit-friendly summaries of steps.\n"
            "â€¢ Cite sources briefly when using web tools.\n"
            f"â€¢ {style_line}\n\n"
            "â€” Harmony I/O Protocol â€”\n"
            "â€¢ To call a tool, emit EXACTLY:\n"
            "  <|channel|>commentary to=functions.TOOLNAME\n"
            "  <|message|>{JSON_ARGS}<|call|>\n"
            "â€¢ Always finish with:\n"
            "  <|channel|>final\n"
            "  <|message|>YOUR ANSWER HERE<|end|>\n\n"
            "â€” Tool Use Policy â€”\n"
            "â€¢ Prefer minimal calls that answer the question directly.\n"
            "â€¢ Summarize tool results; avoid dumping raw blobs unless asked.\n\n"
            "â€” Formatting â€”\n"
            "â€¢ Use clear paragraphs or short bullets when synthesizing.\n"
        )
        # Preserve behavior: fewshots off by default; include only when explicitly enabled
        if getattr(self, "fewshots", False):
            base = base + "\n" + self.few_shots_block()
        return base

    def deepseek_system_prompt(self) -> str:
        """Minimal, neutral system prompt tailored for DeepSeek Chat (v3.x).

        Avoids Harmony-specific markup and focuses on clarity and focus.
        Controlled by PromptManager.verbosity similar to the default prompt.
        """
        verbosity = (self.verbosity or "concise").lower()
        style_line = "Be thorough and structured." if verbosity == "detailed" else "Be concise but complete."
        return (
            "You are a helpful AI assistant.\n"
            f"â€¢ Reasoning effort: {self.reasoning}.\n"
            f"â€¢ Style: {style_line}\n"
            "â€¢ Stay focused on the user's request.\n"
            "â€¢ Use tools only when necessary and summarize results clearly.\n"
            "â€¢ Provide clear steps or short bullets when appropriate.\n"
        )

    # ---------- Post-Tool Reprompt ----------
    def reprompt_after_tools(self) -> str:
        verbose = bool(getattr(self, "verbose_after_tools", False))
        if verbose:
            return (
                "Using the tool results above, produce <|channel|>final with:\n"
                "1) What you checked (1â€“2 sentences)\n"
                "2) Key findings (3â€“7 bullets with brief inline citations)\n"
                "3) Direct answer\n"
                "4) Important caveats\n"
            )
        return (
            "Based on the tool results above, produce <|channel|>final with a clear, synthesized answer. "
            "Summarize; avoid copying raw tool output."
        )

    # ---------- Post-Tool Reprompt (override for cited synthesis) ----------
    def reprompt_after_tools(self) -> str:  # type: ignore[override]
        return (
            "Synthesize an answer only from context.docs and citations above. "
            "Use inline [n] that map to citations[n]. If a claim isn’t supported, say so."
        )

    # ---------- Mem0 Context Block ----------
    @staticmethod
    def mem0_prefix() -> str:
        return "Previous context from user history (use if relevant):"

    @classmethod
    def mem0_prefixes(cls) -> List[str]:
        # Include backward-compatible prefixes to support trimming and cleanup
        return [
            cls.mem0_prefix(),
            "Relevant information:",
            "Relevant user memories",
        ]

    def mem0_context_block(self, bullets: List[str]) -> str:
        prefix = self.mem0_prefix()
        items = "\n".join(bullets)
        tail = "\n\nIntegrate this context naturally into your response only where it adds value."
        return f"{prefix}\n{items}{tail}"

    # ---------- Helpers & Few-shots ----------
    def _flag(self, name: str, default: bool) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        return str(v).strip().lower() in {"1", "true", "yes", "on"}

    def few_shots_block(self) -> str:
        """Optional few-shot guide appended to the system prompt when PROMPT_FEWSHOTS is enabled."""
        return (
            "Examples:\n"
            "- User: 'Weather in Tokyo and 2^10'\n"
            "  Plan: use get_current_weather(city='Tokyo', unit=celsius) then calculate_math(expression='2^10').\n"
            "  Answer: provide a concise sentence combining both results.\n"
            "- User: 'Who is Ada Lovelace?'\n"
            "  Plan: use wikipedia_search(query='Ada Lovelace') then web_fetch(url=<best result URL>) if needed.\n"
            "  Answer: summarize key facts and cite the page briefly.\n"
            "- User: 'Show python files in this folder'\n"
            "  Plan: use list_files(directory='.', extension='.py').\n"
            "  Answer: present a short list of filenames."
        )
