#!/usr/bin/env python
"""Test script for Harmony tool calling functionality."""

import asyncio
import os
import sys
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pytest
from src.application.chat_service import ChatService
from src.infrastructure.ollama.client import OllamaAdapter
from src.infrastructure.tools.registry import DefaultToolRegistry
from src.domain.services.tool_orchestrator import ToolOrchestrator
from src.domain.services.conversation_service import ConversationService
from src.domain.models.conversation import ConversationContext
from src.domain.models.tool import ToolSchema, ToolExecutionContext, ToolCall, ToolResult
from src.domain.interfaces.tool_plugin import ToolPlugin
from src.plugin_loader import PluginManager

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load .env (if present) so OLLAMA_API_KEY is available in tests
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=False)
    # If key is still missing or empty, force override from .env
    if not os.getenv('OLLAMA_API_KEY'):
        load_dotenv(find_dotenv(), override=True)
except Exception:
    # Fallback: search for .env in current and parent directories (up to 5 levels)
    cur = Path(__file__).resolve().parent
    candidate = None
    for _ in range(5):
        p = cur / ".env"
        if p.exists():
            candidate = p
            break
        cur = cur.parent
    if candidate and candidate.exists():
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if not os.getenv(k):
                os.environ[k] = v

# Require API key for this integration test
if not os.getenv('OLLAMA_API_KEY'):
    pytest.skip("Integration test requires OLLAMA_API_KEY and network access", allow_module_level=True)

# Strengthen retry behavior for Turbo (mitigate transient 5xx like 502)
os.environ.setdefault('CLI_MAX_RETRIES', '5')
os.environ.setdefault('CLI_RETRY_BACKOFF_BASE', '1.5')
os.environ.setdefault('CLI_RETRY_JITTER_MAX', '0.2')

async def _test_tool_calling():
    """Test Harmony tool calling with full orchestration."""
    
    # Initialize Ollama client
    api_key = os.getenv('OLLAMA_API_KEY')
    
    # Create a simple weather tool
    class WeatherToolPlugin(ToolPlugin):
        """Mock weather tool plugin for testing."""
        
        def get_schema(self) -> ToolSchema:
            """Return tool schema as ToolSchema."""
            return ToolSchema(
                name="get_current_weather",
                description="Get the current weather in a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "fahrenheit"
                        }
                    },
                    "required": ["location"]
                },
            )
        
        def is_available(self) -> bool:
            return True
        
        async def execute(self, tool_call: ToolCall, context: ToolExecutionContext) -> ToolResult:
            """Execute the weather tool."""
            args = tool_call.arguments or {}
            location = args.get('location', 'Unknown')
            unit = args.get('unit', 'fahrenheit')
            
            # Mock weather data
            temp = 72 if unit == "fahrenheit" else 22
            content = json.dumps({
                "location": location,
                "temperature": temp,
                "unit": unit,
                "conditions": "Sunny",
                "humidity": "45%"
            })
            return ToolResult(tool_call=tool_call, success=True, content=content)

    class CalculatorToolPlugin(ToolPlugin):
        """Mock calculator tool plugin for testing."""
        
        def get_schema(self) -> ToolSchema:
            """Return tool schema as ToolSchema."""
            return ToolSchema(
                name="calculate",
                description="Perform mathematical calculations",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                },
            )
        
        def is_available(self) -> bool:
            return True
        
        async def execute(self, tool_call: ToolCall, context: ToolExecutionContext) -> ToolResult:
            """Execute the calculator tool."""
            args = tool_call.arguments or {}
            expression = args.get('expression', '')
            
            try:
                # Simple eval for demo - NOT FOR PRODUCTION
                result = eval(expression)
                content = json.dumps({"result": result, "expression": expression})
                return ToolResult(tool_call=tool_call, success=True, content=content)
            except Exception as e:
                content = json.dumps({"error": str(e), "expression": expression})
                return ToolResult(tool_call=tool_call, success=False, content=content, error=str(e))

    # Create tool registry and register tools
    tool_registry = DefaultToolRegistry()
    tool_registry.register_tool(WeatherToolPlugin())
    tool_registry.register_tool(CalculatorToolPlugin())
    
    # Initialize services
    llm_client = OllamaAdapter(
        api_key=api_key,
        model="gpt-oss:120b",
        host="https://ollama.com",
        logger=logger
    )
    
    conversation_service = ConversationService()
    tool_orchestrator = ToolOrchestrator(tool_registry)
    
    chat_service = ChatService(
        llm_client=llm_client,
        conversation_service=conversation_service,
        tool_orchestrator=tool_orchestrator
    )
    
    # Create context with tools enabled
    context = ConversationContext(
        model="gpt-oss:120b",
        enable_tools=True,
        reasoning="high",
    )
    # Warm-up call (no tools) to reduce cold-start 5xx on Turbo
    try:
        warmup_ctx = ConversationContext(model="gpt-oss:120b", enable_tools=False, max_output_tokens=16)
        _ = llm_client.chat([
            {"role": "user", "content": "Warm up"}
        ], warmup_ctx)
    except Exception as _warmup_err:
        logger.debug(f"Warm-up chat failed (continuing): {_warmup_err}")
    
    # Test queries
    test_queries = [
        "What's the weather like in San Francisco?",
        "Calculate 42 * 17 + 8",
        "What's the weather in both Miami and Toronto?"
    ]
    
    conversation_id = "test-harmony-" + str(os.getpid())
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Send chat request
        result = await chat_service.chat(
            message=query,
            context=context,
            conversation_id=conversation_id
        )
        
        print(f"\nSuccess: {result.success}")
        print(f"\nResponse:\n{result.content}")
        
        if result.metadata:
            print(f"\nMetadata:")
            for key, value in result.metadata.items():
                if key != 'message':  # Skip raw message for brevity
                    print(f"  {key}: {value}")
        
        if result.error:
            print(f"\nError: {result.error}")

def test_tool_calling():
    asyncio.run(_test_tool_calling())

if __name__ == "__main__":
    asyncio.run(_test_tool_calling())
