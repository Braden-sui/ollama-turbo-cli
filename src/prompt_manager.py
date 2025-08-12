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
            "- Gather all necessary information before formulating your response\n"
            "- You may call multiple tools if needed to fully answer the question\n"
            "- After gathering data, synthesize a complete answer\n"
            "- Avoid redundant tool calls - use each tool at most once per request"
        )

        parts.append(
            "Tool selection strategy:\n"
            "- get_current_weather: for meteorological queries\n"
            "- calculate_math: for any mathematical computation\n"
            "- web_fetch: for retrieving specific URL content\n"
            "- duckduckgo_search: for general web searches or recent info\n"
            "- wikipedia_search: for encyclopedic information\n"
            "- system_info: for local system details\n"
            "- list_files: for directory exploration"
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
