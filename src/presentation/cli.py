"""
CLI presentation layer - Clean interface for command-line interactions.
Coordinates with application services to provide user interface.
"""

from __future__ import annotations
import asyncio
import logging
import sys
import os
from typing import Optional

from ..application.chat_service import ChatService
from ..domain.models.conversation import ConversationContext
from ..domain.models.memory import MemoryContext
from ..infrastructure.config.settings import get_settings


class ChatCLI:
    """CLI interface for chat interactions."""
    
    def __init__(
        self,
        chat_service: ChatService,
        logger: Optional[logging.Logger] = None
    ):
        self._chat_service = chat_service
        self._logger = logger or logging.getLogger(__name__)
        self._settings = get_settings()
        self._conversation_id = "main"
        self._memory_context: Optional[MemoryContext] = None
        
        # Initialize memory context if available
        if self._settings.memory_enabled:
            # Resolve effective user_id with env fallback (ensures backend identity like 'Braden')
            user_id = self._settings.memory.user_id
            env_user = os.getenv('MEM0_USER_ID')
            if (not user_id or user_id == 'default-user') and env_user:
                user_id = env_user
            self._memory_context = MemoryContext(
                user_id=user_id,
                agent_id="ollama-turbo-cli",
                metadata={"source": "cli", "version": "1.0"}
            )
    
    def interactive_mode(self) -> None:
        """Run interactive chat mode."""
        if not self._settings.quiet:
            self._print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle CLI commands
                if self._handle_cli_commands(user_input):
                    continue
                
                # Handle memory commands
                if self._handle_memory_commands(user_input):
                    continue
                
                # Regular chat interaction
                self._process_chat_message(user_input, stream=True)
                
            except KeyboardInterrupt:
                if not self._settings.quiet:
                    print("\n\n⚠️ Use 'quit' or 'exit' to leave the chat")
                continue
            except EOFError:
                break
            except Exception as e:
                self._logger.error(f"Interactive mode error: {e}")
                if not self._settings.quiet:
                    print(f"\n❌ Error: {str(e)}")
                continue
        
        if not self._settings.quiet:
            print("👋 Goodbye!")
    
    def single_message(self, message: str, stream: bool = False) -> str:
        """Process a single message and return response."""
        try:
            return self._process_chat_message(message, stream=stream)
        except Exception as e:
            self._logger.error(f"Single message error: {e}")
            return f"Error: {e}"
    
    def _print_welcome(self) -> None:
        """Print welcome message with service status."""
        print("🚀 Ollama Turbo CLI - Interactive Mode")
        print(f"📝 Model: {self._settings.ollama.model}")
        print(f"🔧 Tools: {'Enabled' if self._settings.tools_enabled else 'Disabled'}")
        
        if self._settings.memory_enabled:
            effective_user = getattr(self._memory_context, 'user_id', None) or self._settings.memory.user_id
            print(f"🧠 Memory: Enabled (user: {effective_user})")
        else:
            print("🧠 Memory: Disabled (no API key)")
        
        print("💡 Commands: 'quit'/'exit' to exit, 'clear' to clear history, 'history' to show history, '/mem ...' for memory ops")
        print("-" * 60)
    
    def _handle_cli_commands(self, user_input: str) -> bool:
        """Handle built-in CLI commands."""
        command = user_input.lower()
        
        if command in ['quit', 'exit']:
            sys.exit(0)
        elif command == 'clear':
            self._chat_service.clear_conversation(self._conversation_id)
            if not self._settings.quiet:
                print("✅ History cleared")
            return True
        elif command == 'history':
            history = self._chat_service.get_conversation_history(self._conversation_id)
            if history and not self._settings.quiet:
                print("\n📜 Conversation History:")
                print(history)
            return True
        
        return False
    
    def _handle_memory_commands(self, user_input: str) -> bool:
        """Handle memory-related commands."""
        if not user_input.startswith('/mem') and not self._memory_context:
            return False
        
        try:
            if user_input.startswith('/mem '):
                # Extract command after /mem
                command = user_input[5:].strip()
                response = asyncio.run(
                    self._chat_service.process_memory_command(command, self._memory_context)
                )
                if response:
                    print(response)
                return True
            
            # Natural language memory patterns
            if self._memory_context:
                response = asyncio.run(
                    self._chat_service.process_memory_command(user_input, self._memory_context)
                )
                if response:
                    print(response)
                    return True
            
        except Exception as e:
            self._logger.debug(f"Memory command error: {e}")
        
        return False
    
    def _process_chat_message(self, message: str, stream: bool = False) -> str:
        """Process chat message and return response."""
        # Create conversation context
        context = ConversationContext(
            model=self._settings.ollama.model,
            enable_tools=self._settings.tools_enabled,
            stream=stream,
            max_output_tokens=self._settings.ollama.max_output_tokens,
            ctx_size=self._settings.ollama.ctx_size,
            reasoning=self._settings.conversation.reasoning,
            show_trace=self._settings.show_trace,
            quiet=self._settings.quiet
        )
        
        if stream:
            return self._process_streaming_chat(message, context)
        else:
            return self._process_non_streaming_chat(message, context)
    
    def _process_streaming_chat(self, message: str, context: ConversationContext) -> str:
        """Process streaming chat interaction."""
        if not self._settings.quiet:
            print("Assistant: ", end="", flush=True)
        
        accumulated_response = ""
        
        try:
            # Use asyncio to handle the streaming
            async def stream_chat():
                nonlocal accumulated_response
                async for chunk in self._chat_service.chat_stream(
                    message, context, self._conversation_id, self._memory_context
                ):
                    accumulated_response += chunk
                    if not self._settings.quiet:
                        print(chunk, end="", flush=True)
            
            asyncio.run(stream_chat())
            
            if not self._settings.quiet:
                print()  # New line after streaming
            
            return accumulated_response
            
        except Exception as e:
            self._logger.error(f"Streaming chat error: {e}")
            error_response = f"Streaming error: {e}"
            if not self._settings.quiet:
                print(error_response)
            return error_response
    
    def _process_non_streaming_chat(self, message: str, context: ConversationContext) -> str:
        """Process non-streaming chat interaction."""
        try:
            result = asyncio.run(
                self._chat_service.chat(
                    message, context, self._conversation_id, self._memory_context
                )
            )
            
            if result.success:
                if not self._settings.quiet:
                    print(f"Assistant: {result.content}")
                return result.content
            else:
                error_msg = f"Chat error: {result.error}"
                if not self._settings.quiet:
                    print(f"❌ {error_msg}")
                return error_msg
                
        except Exception as e:
            self._logger.error(f"Non-streaming chat error: {e}")
            error_response = f"Chat error: {e}"
            if not self._settings.quiet:
                print(f"❌ {error_response}")
            return error_response
