"""
Harmony format utilities for tool definitions and messages.
Converts tool schemas to TypeScript-like format as per OpenAI Harmony specification.
"""

from typing import List, Dict, Any, Optional
import json


class HarmonyFormatter:
    """Formatter for OpenAI Harmony protocol."""
    
    @staticmethod
    def format_tools_as_typescript(tools: List[Dict[str, Any]]) -> str:
        """
        Convert tool schemas to TypeScript-like format for Harmony.
        
        Args:
            tools: List of tool schemas in OpenAI format
            
        Returns:
            TypeScript-formatted tool definitions
        """
        if not tools:
            return ""
        
        lines = ["# Tools", "## functions", "namespace functions {"]
        
        for tool in tools:
            if tool.get("type") != "function":
                continue
                
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})
            
            # Add description as JSDoc for better editor support
            if description:
                lines.append(f"  /** {description} */")
            
            # Format function signature
            if not parameters or not parameters.get("properties"):
                # No parameters
                lines.append(f"  type {name} = () => any;")
            else:
                # Has parameters - format as inline type
                lines.append(f"  type {name} = (_: {{")
                
                properties = parameters.get("properties", {})
                required = parameters.get("required", [])
                
                for prop_name, prop_schema in properties.items():
                    prop_desc = prop_schema.get("description", "")
                    prop_type = HarmonyFormatter._convert_json_type_to_typescript(prop_schema)
                    
                    # Add description as JSDoc if present
                    if prop_desc:
                        lines.append(f"    /** {prop_desc} */")
                    
                    # Add property with optional marker if not required
                    optional = "?" if prop_name not in required else ""
                    
                    # Handle enums
                    if "enum" in prop_schema:
                        enum_values = " | ".join([f'"{v}"' for v in prop_schema["enum"]])
                        lines.append(f"    {prop_name}{optional}: {enum_values},")
                        if "default" in prop_schema:
                            lines.append(f"    // default: {prop_schema['default']}")
                    else:
                        lines.append(f"    {prop_name}{optional}: {prop_type},")
                
                lines.append("  }) => any;")
            
            lines.append("")  # Empty line between functions
        
        lines.append("} // namespace functions")
        
        return "\n".join(lines)
    
    @staticmethod
    def _convert_json_type_to_typescript(schema: Dict[str, Any]) -> str:
        """Convert JSON schema type to TypeScript type."""
        json_type = schema.get("type", "any")
        
        type_mapping = {
            "string": "string",
            "number": "number",
            "integer": "number",
            "boolean": "boolean",
            "array": "any[]",  # Could be more specific with items schema
            "object": "object",
            "null": "null"
        }
        
        # Handle arrays with specific item types
        if json_type == "array" and "items" in schema:
            item_type = HarmonyFormatter._convert_json_type_to_typescript(schema["items"])
            return f"{item_type}[]"
        
        return type_mapping.get(json_type, "any")
    
    @staticmethod
    def format_tool_call_message(tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Format a tool call in Harmony format.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Formatted tool call message
        """
        args_json = json.dumps(arguments, separators=(',', ':'), ensure_ascii=False)
        return (
            f"<|start|>assistant"
            f"<|channel|>commentary to=functions.{tool_name} <|constrain|>json"
            f"<|message|>{args_json}<|call|>"
            f"<|end|>"
        )
    
    @staticmethod
    def format_tool_response_message(tool_name: str, result: str) -> str:
        """
        Format a tool response in Harmony format.
        
        Args:
            tool_name: Name of the tool that was called
            result: Result from the tool execution
            
        Returns:
            Formatted tool response message
        """
        # Ensure result is JSON-serializable and safe for Harmony tokens
        if not isinstance(result, str):
            try:
                result = json.dumps(result, separators=(',', ':'), ensure_ascii=False)
            except Exception:
                result = str(result)

        safe = HarmonyFormatter._sanitize_harmony_text(result)
        return f"<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>{safe}<|end|>"

    @staticmethod
    def _sanitize_harmony_text(text: str) -> str:
        """Escape Harmony tag-like tokens inside message content to avoid collisions."""
        if not isinstance(text, str):
            return text
        # Minimal escaping to prevent accidental tag boundaries in content
        return text.replace("<|", "<\\|").replace("|>", "\\|>")
    
    @staticmethod
    def extract_tool_calls_from_harmony(content: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from Harmony-formatted content using a token scanner.

        This is more robust than regex against whitespace, optional tags, and
        JSON that contains braces inside strings.
        """
        tool_calls: List[Dict[str, Any]] = []
        if not content:
            return tool_calls

        CHANNEL = "<|channel|>"
        MESSAGE = "<|message|>"
        CALL = "<|call|>"

        idx = 0
        n = len(content)

        while idx < n:
            start = content.find(CHANNEL, idx)
            if start == -1:
                break

            # Find the end of header (up to <|message|>)
            header_start = start + len(CHANNEL)
            msg_tag = content.find(MESSAGE, header_start)
            if msg_tag == -1:
                break  # incomplete

            header = content[start:msg_tag]
            header_lower = header.lower()
            if "commentary" not in header_lower:
                idx = msg_tag + len(MESSAGE)
                continue

            # Look for to=functions.NAME inside header
            to_key = "to=functions."
            to_pos = header_lower.find(to_key)
            if to_pos == -1:
                idx = msg_tag + len(MESSAGE)
                continue

            name_start = start + to_pos + len(to_key)
            # Extract name consisting of [A-Za-z0-9_ .] (dot and underscore allowed)
            j = name_start
            while j < n and content[j] not in " \n\t\r<|":
                j += 1
            tool_name = content[name_start:j]
            if not tool_name:
                idx = msg_tag + len(MESSAGE)
                continue

            # Extract arguments between <|message|> and <|call|>
            call_pos = content.find(CALL, msg_tag + len(MESSAGE))
            if call_pos == -1:
                # Incomplete; advance to after MESSAGE to continue scanning
                idx = msg_tag + len(MESSAGE)
                continue

            args_str = content[msg_tag + len(MESSAGE):call_pos]
            args_str = args_str.strip()
            try:
                arguments = json.loads(args_str)
                tool_calls.append({"name": tool_name, "arguments": arguments})
            except Exception:
                # If JSON fails, skip this candidate and continue
                pass

            idx = call_pos + len(CALL)

        return tool_calls
    
    @staticmethod
    def format_system_prompt_with_tools(tools: List[Dict[str, Any]], 
                                       reasoning: str = "high",
                                       custom_instructions: str = "") -> str:
        """
        Format complete system prompt with tools in Harmony format.
        
        Args:
            tools: List of tool schemas
            reasoning: Reasoning level (high/low/auto)
            custom_instructions: Additional custom instructions
            
        Returns:
            Complete formatted system prompt
        """
        from datetime import datetime
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # System message with proper line formatting (per Harmony examples)
        system_lines = [
            "You are ChatGPT, a large language model trained by OpenAI.",
            "Knowledge cutoff: 2024-06",
            f"Current date: {current_date}",
            f"reasoning: {reasoning}",  # lowercase key on its own line
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.",
        ]
        if tools:
            system_lines.append("Calls to these tools must go to the commentary channel: 'functions'.")
        system_msg = "\n".join(system_lines)
        
        # Note: tools routing line already included above when tools are present.
        
        # Developer message with tools
        developer_parts = ["# Instructions"]
        
        if custom_instructions:
            developer_parts.append(custom_instructions)
        else:
            developer_parts.append("You are a helpful AI assistant with access to tools.")
        
        # Add tools in TypeScript format
        if tools:
            developer_parts.append("")
            developer_parts.append(HarmonyFormatter.format_tools_as_typescript(tools))
            developer_parts.append("")
            developer_parts.append("# Tool Calling (Harmony)")
            developer_parts.append("- When a tool is needed, do not narrate your plan. Emit exactly one Harmony commentary tool call and nothing else.")
            developer_parts.append("- Format for a tool call (no extra text before/after):")
            developer_parts.append(
                '  <|start|>assistant'
                '<|channel|>commentary to=functions.TOOL_NAME <|constrain|>json'
                '<|message|>{"arg":"value"}<|call|><|end|>'
            )
            developer_parts.append("")
            developer_parts.append("# Final Answer (Harmony)")
            developer_parts.append("- After receiving tool responses (if any), emit exactly one assistant message on the final channel.")
            developer_parts.append("- Format for the final answer (no extra text before/after):")
            developer_parts.append(
                '  <|start|>assistant<|channel|>final<|message|>Your answer here<|end|>'
            )
            developer_parts.append("- Do not emit analysis or commentary alongside the final answer. The final answer must be the only content in the final channel block.")
            developer_parts.append("- Every assistant emission must include a Harmony channel header. Do not output plain text responses without a channel.")
            developer_parts.append("- If the model uses a tool_calls field instead of commentary, its items must be of shape: [{\"name\",\"arguments\"}]. Do NOT use OpenAI fields (no type, no function sub-object).")
            developer_parts.append("- Never output raw JSON tool arguments to the user; only use the Harmony tool-call or the final natural-language answer.")
        
        developer_msg = "\n".join(developer_parts)
        
        return f"<|start|>system<|message|>{system_msg}<|end|><|start|>developer<|message|>{developer_msg}<|end|>"
