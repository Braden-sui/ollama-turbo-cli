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


class PromptManager:
    """Builds and returns prompt strings used across the client."""

    def __init__(self, reasoning: str = "high") -> None:
        self.reasoning = reasoning if reasoning in {"low", "medium", "high"} else "high"
        self.version = os.getenv("PROMPT_VERSION", "v1").strip() or "v1"

    # ---------- System / Initial Prompt ----------
    def initial_system_prompt(self) -> str:
        """Return the initial system directive used to steer model behavior."""
        # Sections can be toggled in the future via env flags if needed
        parts: List[str] = []
        parts.append(f"You are a helpful assistant with access to tools. Reasoning level: {self.reasoning}.")

        parts.append(
            "Guidelines:\n"
            "- Analyze the user's request to determine if tools are needed\n"
            "- For factual queries requiring current data, use appropriate tools\n"
            "- For computational tasks, use the calculator tool\n"
            "- Keep your thinking process internal - only show final answers\n"
            "- When using web tools, cite sources briefly\n"
            "- Be concise but complete in your responses"
        )

        parts.append(
            "Tool usage strategy:\n"
            "- Gather necessary facts via tools before composing your answer\n"
            "- Chain multiple tools only when each adds required information\n"
            "- Synthesize a final textual answer after tools\n"
            "- Avoid redundant or speculative tool calls"
        )

        if self._flag("PROMPT_INCLUDE_TOOL_GUIDE", True):
            parts.append(
                "Tool selection guide:\n"
                "- get_current_weather: current conditions for a specific city (no 'here'); respects unit; not for forecasts.\n"
                "- calculate_math: pure math expressions; no units or natural language.\n"
                "- duckduckgo_search: keyless web search for sources or quick facts; keep results small.\n"
                "- wikipedia_search: canonical topic pages; then use web_fetch to read details.\n"
                "- web_fetch: read a specific HTTPS URL with minimal bytes; only compact summaries are injected.\n"
                "- execute_shell: read-only diagnostics in a sandbox; disabled by default; allowlist only.\n"
                "- list_files: inspect a directory; prefer extension filter to reduce noise.\n"
                "- get_system_info: environment snapshot; call only when needed."
            )

        parts.append(
            "Error handling:\n"
            "- If a tool fails, acknowledge the limitation and offer alternatives\n"
            "- Never expose technical error details to the user\n"
            "- Suggest workarounds when appropriate"
        )

        parts.append(
            "Conversation context:\n"
            "- Focus on the most recent exchanges\n"
            "- Reference earlier context only when directly relevant\n"
            "- Maintain continuity without repeating information"
        )

        parts.append(
            "Shell command guidelines:\n"
            "- Only execute read-only, safe commands\n"
            "- Never run commands that modify system state\n"
            "- Explain what each command does before execution\n"
            "- Prefer built-in tools over shell commands when possible"
        )
        if self._flag("PROMPT_FEWSHOTS", False):
            parts.append(self.few_shots_block())

        return "\n".join(parts)

    # ---------- Post-Tool Reprompt ----------
    def reprompt_after_tools(self) -> str:
        return (
            "Based on the tool results above, provide a comprehensive answer to the user's original question. "
            "Synthesize the information naturally without repeating raw tool output."
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
