"""
Ollama Turbo client implementation with tool calling and streaming support.
"""

import json
import sys
import logging
import os
from typing import Dict, Any, List, Optional, Union
import ollama
from ollama import Client

from .plugin_loader import TOOL_SCHEMAS, TOOL_FUNCTIONS
from .utils import with_retry, RetryableError, OllamaAPIError, truncate_text, format_conversation_history


class OllamaTurboClient:
    """Client for interacting with gpt-oss:120b via Ollama Turbo."""
    
    def __init__(self, api_key: str, model: str = "gpt-oss:120b", enable_tools: bool = True, show_trace: bool = False, reasoning: str = "high", quiet: bool = False, max_output_tokens: Optional[int] = None, ctx_size: Optional[int] = None, tool_print_limit: int = 200):
        """Initialize Ollama Turbo client.
        
        Args:
            api_key: Ollama API key for authentication
            model: Model name to use (default: gpt-oss:120b)
            enable_tools: Whether to enable tool calling capabilities
            show_trace: Whether to collect and print a separated reasoning trace
            reasoning: Reasoning effort directive ('low' | 'medium' | 'high')
            quiet: Reduce CLI noise (suppress helper prints)
            max_output_tokens: Limit on tokens to generate (maps to options.num_predict)
            ctx_size: Context window size (maps to options.num_ctx)
            tool_print_limit: CLI print truncation for tool outputs (characters)
        """
        self.api_key = api_key
        self.model = model
        self.enable_tools = enable_tools
        self.show_trace = show_trace
        self.quiet = quiet
        self.reasoning = reasoning if reasoning in {"low", "medium", "high"} else "high"
        self.trace: List[str] = []
        self.logger = logging.getLogger(__name__)
        self.max_output_tokens = max_output_tokens
        self.ctx_size = ctx_size
        self.tool_print_limit = tool_print_limit
        
        # Initialize Ollama client with authentication
        # Note: Ollama Turbo uses Authorization header without 'Bearer' prefix
        self.client = Client(
            host='https://ollama.com',
            headers={'Authorization': api_key}
        )
        
        # Initialize conversation history with a system directive for reasoning effort
        self.conversation_history = [
            {
                'role': 'system',
                'content': (
                    f"Reasoning effort: {self.reasoning}. "
                    "Use tools when helpful. Do not expose chain-of-thought. Provide a direct answer, and include brief source URLs when tools are used."
                )
            }
        ]
        self.max_history = int(os.getenv('MAX_CONVERSATION_HISTORY', '50'))
        
        # Set up tools if enabled
        if enable_tools:
            self.tools = TOOL_SCHEMAS
            self.tool_functions = TOOL_FUNCTIONS
        else:
            self.tools = []
            self.tool_functions = {}
        
        self.logger.info(f"Initialized client with model: {model}, tools enabled: {enable_tools}, reasoning={self.reasoning}, quiet={self.quiet}")
        # Initial trace state
        if self.show_trace:
            self.trace.append(f"client:init model={model} tools={'on' if enable_tools else 'off'} reasoning={self.reasoning} quiet={'on' if self.quiet else 'off'}")

    def _trace(self, event: str):
        """Record a structured, non-sensitive trace event."""
        if self.show_trace:
            self.trace.append(event)

    def _print_trace(self):
        """Print a separated reasoning trace section."""
        if self.show_trace and self.trace:
            # Send to stderr to avoid mixing with streamed stdout
            print("\n\n--- Reasoning Trace ---", file=sys.stderr, flush=True)
            for item in self.trace:
                print(f" • {item}", file=sys.stderr, flush=True)
            # Clear after printing to avoid duplication on next call
            self.trace = []
    
    def chat(self, message: str, stream: bool = False) -> str:
        """Send a message to the model and get a response.
        
        Args:
            message: User message to send
            stream: Whether to stream the response
            
        Returns:
            Model response as string
        """
        # Reset trace for this turn
        self.trace = [] if self.show_trace else []
        self._trace(f"chat:start stream={'on' if stream else 'off'}")

        # Add user message to history
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
        
        # Trim history if needed
        self._trim_history()
        
        try:
            if stream:
                result = self._handle_streaming_chat()
            else:
                result = self._handle_standard_chat()
            # Print separated trace after output
            self._print_trace()
            return result
        except Exception as e:
            self.logger.error(f"Chat error: {e}")
            error_msg = f"Error during chat: {str(e)}"
            self.conversation_history.append({
                'role': 'assistant',
                'content': error_msg
            })
            self._trace(f"chat:error {type(e).__name__}")
            self._print_trace()
            return error_msg
    
    @with_retry(max_retries=3)
    def _handle_standard_chat(self) -> str:
        """Handle non-streaming chat interaction."""
        try:
            # Make API call with tools if enabled
            kwargs = {
                'model': self.model,
                'messages': self.conversation_history,
            }
            # Generation options
            options = {}
            if self.max_output_tokens is not None:
                options['num_predict'] = self.max_output_tokens
            if self.ctx_size is not None:
                options['num_ctx'] = self.ctx_size
            if options:
                kwargs['options'] = options
            
            if self.enable_tools and self.tools:
                kwargs['tools'] = self.tools
            
            self._trace("request:standard")
            response = self.client.chat(**kwargs)
            
            # Extract message from response
            message = response.get('message', {})
            content = message.get('content', '')
            tool_calls = message.get('tool_calls', [])
            
            # Process tool calls if present
            if tool_calls and self.enable_tools:
                # Add assistant message with tool calls
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': content,
                    'tool_calls': tool_calls
                })
                names = [tc.get('function', {}).get('name') for tc in tool_calls]
                self._trace(f"tools:detected {len(tool_calls)} -> {', '.join(n for n in names if n)}")
                
                # Execute tools
                tool_results = self._execute_tool_calls(tool_calls)
                self._trace(f"tools:executed {len(tool_results)}")
                
                # Add tool results to history
                self.conversation_history.append({
                    'role': 'tool',
                    'content': '\n'.join(tool_results)
                })
                
                # Reprompt model to synthesize an answer using tool details
                self._trace("reprompt:after-tools")
                self.conversation_history.append({
                    'role': 'user',
                    'content': "Please use these details to answer the user's original question."
                })
                
                # Get final response after tool execution
                self._trace("request:final-after-tools")
                # Final response after tools with same options
                final_kwargs = {
                    'model': self.model,
                    'messages': self.conversation_history,
                }
                # Do not include tools in the final call; force a textual answer
                if options:
                    final_kwargs['options'] = options
                final_response = self.client.chat(**final_kwargs)
                
                final_content = final_response.get('message', {}).get('content', '')
                
                # Add final response to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': final_content
                })
                
                # Return combined response
                return f"{content}\n\n[Tool Results]\n" + '\n'.join(tool_results) + f"\n\n{final_content}"
            else:
                # No tool calls, just add response to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': content
                })
                self._trace("tools:none")
                return content
                
        except Exception as e:
            self.logger.error(f"Standard chat error: {e}")
            self._trace(f"standard:error {type(e).__name__}")
            raise RetryableError(f"API request failed: {e}")
    
    def _handle_streaming_chat(self) -> str:
        """Handle streaming chat interaction with tool support."""
        return self.handle_streaming_response(
            self._create_streaming_response(),
            tools_enabled=self.enable_tools
        )
    
    @with_retry(max_retries=3)
    def _create_streaming_response(self):
        """Create a streaming response from the API."""
        try:
            kwargs = {
                'model': self.model,
                'messages': self.conversation_history,
                'stream': True
            }
            # Generation options
            options = {}
            if self.max_output_tokens is not None:
                options['num_predict'] = self.max_output_tokens
            if self.ctx_size is not None:
                options['num_ctx'] = self.ctx_size
            if options:
                kwargs['options'] = options
            
            if self.enable_tools and self.tools:
                kwargs['tools'] = self.tools
            
            self._trace("request:stream:start")
            return self.client.chat(**kwargs)
        except Exception as e:
            self.logger.error(f"Streaming creation error: {e}")
            self._trace(f"stream:init:error {type(e).__name__}")
            raise RetryableError(f"Failed to create streaming response: {e}")
    
    def handle_streaming_response(self, response_stream, tools_enabled: bool = True) -> str:
        """Complete streaming response handler with tool call support."""
        full_content = ""
        tool_calls = []
        
        # For cleaner UX: if tools are enabled, buffer initial stream and only print
        # the final answer after tools (to avoid duplicate messages). If tools are
        # disabled, stream directly to stdout.
        if not self.quiet and not tools_enabled:
            print("🤖 Assistant: ", end="", flush=True)
        
        try:
            for chunk in response_stream:
                message = chunk.get('message', {})
                
                # Handle content streaming
                if message.get('content'):
                    content = message['content']
                    # Stream directly only when tools are not enabled; otherwise buffer
                    if not tools_enabled:
                        print(content, end="", flush=True)
                    full_content += content
                
                # Collect tool calls
                if message.get('tool_calls') and tools_enabled:
                    for tool_call in message['tool_calls']:
                        # Find if this tool call already exists (streaming may send in parts)
                        existing = False
                        for tc in tool_calls:
                            if tc.get('id') == tool_call.get('id'):
                                # Update existing tool call
                                tc.update(tool_call)
                                existing = True
                                break
                        if not existing:
                            tool_calls.append(tool_call)
            
            if not tools_enabled:
                print()  # New line after content
            
            # Process tool calls if any
            if tool_calls:
                if not self.show_trace and not self.quiet:
                    print("\n🔧 Processing tool calls...")
                names = [tc.get('function', {}).get('name') for tc in tool_calls]
                self._trace(f"tools:detected {len(tool_calls)} -> {', '.join(n for n in names if n)}")
                
                # Add assistant message with tool calls to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': full_content,
                    'tool_calls': tool_calls
                })
                
                # Execute tools and collect results
                tool_results = self._execute_tool_calls(tool_calls)
                self._trace(f"tools:executed {len(tool_results)}")
                
                # Add tool results to conversation
                tool_message = {
                    'role': 'tool',
                    'content': '\n'.join(tool_results)
                }
                self.conversation_history.append(tool_message)
                
                # Reprompt model to synthesize an answer using tool details
                self._trace("reprompt:after-tools")
                self.conversation_history.append({
                    'role': 'user',
                    'content': "Please use these details to answer the user's original question."
                })
                
                # Get final response after tool execution
                if not self.quiet:
                    # Print the assistant prefix only for the final visible response
                    print("\n🤖 Assistant: ", end="", flush=True)
                self._trace("request:final-after-tools:stream")
                final_stream_kwargs = {
                    'model': self.model,
                    'messages': self.conversation_history,
                    'stream': True
                }
                # Do not include tools in the final call; force a textual answer
                final_options = {}
                if self.max_output_tokens is not None:
                    final_options['num_predict'] = self.max_output_tokens
                if self.ctx_size is not None:
                    final_options['num_ctx'] = self.ctx_size
                if final_options:
                    final_stream_kwargs['options'] = final_options
                final_response_stream = self.client.chat(**final_stream_kwargs)
                
                final_content = ""
                for chunk in final_response_stream:
                    if chunk.get('message', {}).get('content'):
                        content = chunk['message']['content']
                        print(content, end="", flush=True)
                        final_content += content
                
                print()
                
                # Add final response to history
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': final_content
                })
                
                return full_content + "\n\n[Tool Results]\n" + '\n'.join(tool_results) + "\n\n" + final_content
            
            else:
                # No tool calls
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': full_content
                })
                self._trace("tools:none")
                # If tools are disabled we already streamed to stdout; if enabled, we
                # buffered and should now print once to stdout.
                if tools_enabled:
                    if not self.quiet:
                        print("🤖 Assistant: ", end="", flush=True)
                    if full_content:
                        print(full_content)
                return full_content
                
        except KeyboardInterrupt:
            print("\n⚠️ Streaming interrupted by user")
            self._trace("stream:interrupted")
            return full_content + "\n[Interrupted]"
        except Exception as e:
            # Fallback: if streaming fails (e.g., validation errors from client),
            # degrade gracefully to non-streaming standard chat without noisy CLI output.
            self.logger.debug(f"Streaming error encountered; falling back to non-streaming: {e}")
            self._trace("stream:error -> fallback")
            try:
                final = self._handle_standard_chat()
                # Print only the final response so the user still sees a complete answer
                if final:
                    print(final)
                self._trace("fallback:success")
                return (full_content + "\n" + final) if full_content else final
            except Exception as e2:
                # Keep CLI clean; log the details and show a generic message
                self.logger.error(f"Non-streaming fallback also failed: {e2}")
                print("\n❌ Sorry, something went wrong. Please try again.")
                self._trace("fallback:error")
                return full_content + "\n[Error]"
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[str]:
        """Execute tool calls and return results.
        
        Args:
            tool_calls: List of tool call dictionaries
            
        Returns:
            List of tool execution results
        """
        tool_results = []
        
        for i, tool_call in enumerate(tool_calls, 1):
            function_name = tool_call.get('function', {}).get('name')
            function_args = tool_call.get('function', {}).get('arguments', {})
            
            # Handle arguments that might be JSON strings
            if isinstance(function_args, str):
                try:
                    function_args = json.loads(function_args)
                except json.JSONDecodeError:
                    function_args = {}
            
            if not self.show_trace and not self.quiet:
                print(f"   {i}. Executing {function_name}({', '.join(f'{k}={v}' for k, v in function_args.items())})")
            self._trace(f"tool:exec {function_name}")
            
            if function_name in self.tool_functions:
                try:
                    result = self.tool_functions[function_name](**function_args)
                    tool_results.append(result)
                    if not self.show_trace and not self.quiet:
                        print(f"      ✅ Result: {truncate_text(result, self.tool_print_limit)}")
                    self._trace(f"tool:ok {function_name}")
                except Exception as e:
                    error_result = f"Error executing {function_name}: {str(e)}"
                    tool_results.append(error_result)
                    if not self.show_trace and not self.quiet:
                        print(f"      ❌ {error_result}")
                    self._trace(f"tool:error {function_name}")
            else:
                error_result = f"Unknown tool: {function_name}"
                tool_results.append(error_result)
                if not self.show_trace and not self.quiet:
                    print(f"      ❌ {error_result}")
                self._trace(f"tool:unknown {function_name}")
        
        return tool_results
    
    def _trim_history(self):
        """Trim conversation history to maximum size."""
        if len(self.conversation_history) > self.max_history:
            # Keep the first system message if present, and the most recent messages
            if self.conversation_history[0].get('role') == 'system':
                self.conversation_history = [self.conversation_history[0]] + \
                                          self.conversation_history[-(self.max_history-1):]
            else:
                self.conversation_history = self.conversation_history[-self.max_history:]
    
    def clear_history(self):
        """Clear conversation history."""
        # Preserve the system reasoning directive when clearing
        self.conversation_history = [
            {
                'role': 'system',
                'content': (
                    f"Reasoning effort: {self.reasoning}. "
                    "Use tools when helpful. Do not expose chain-of-thought. Output only final answers."
                )
            }
        ]
        self.logger.info("Conversation history cleared")
    
    def get_history(self) -> str:
        """Get formatted conversation history.
        
        Returns:
            Formatted conversation history string
        """
        return format_conversation_history(self.conversation_history)
    
    def interactive_mode(self):
        """Run interactive chat mode."""
        if not self.quiet:
            print("🚀 Ollama Turbo CLI - Interactive Mode")
            print(f"📝 Model: {self.model}")
            print(f"🔧 Tools: {'Enabled' if self.enable_tools else 'Disabled'}")
            print("💡 Commands: 'quit'/'exit' to exit, 'clear' to clear history, 'history' to show history")
            print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    if not self.quiet:
                        print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    if not self.quiet:
                        print("✅ History cleared")
                    continue
                elif user_input.lower() == 'history':
                    if not self.quiet:
                        print("\n📜 Conversation History:")
                        print(self.get_history())
                    continue
                
                # Send message to model
                if not self.quiet:
                    print()  # Empty line for better formatting
                response = self.chat(user_input, stream=True)
                
                # Response is already printed during streaming
                
            except KeyboardInterrupt:
                if not self.quiet:
                    print("\n\n⚠️ Use 'quit' or 'exit' to leave the chat")
                continue
            except Exception as e:
                self.logger.error(f"Interactive mode error: {e}")
                if not self.quiet:
                    print(f"\n❌ Error: {str(e)}")
                continue
