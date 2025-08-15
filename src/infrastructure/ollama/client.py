"""
Ollama client adapter - Infrastructure implementation of LLM client protocol.
Handles communication with Ollama Turbo cloud service.
"""

from __future__ import annotations
import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Iterator
import json

import ollama
from ollama import Client

from ...domain.models.conversation import ConversationResult, ConversationContext, ConversationState
from ...domain.models.tool import ToolSchema
from ...domain.interfaces.llm_client import LLMClient
from .retry import RetryableOllamaClient


class OllamaAdapter(LLMClient):
    """Adapter for Ollama Turbo cloud service implementing LLMClient protocol."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-oss:120b",
        host: str = "https://ollama.com",
        logger: Optional[logging.Logger] = None
    ):
        self._api_key = api_key
        self._model = model
        self._host = host
        self._logger = logger or logging.getLogger(__name__)
        # Detect cloud host to adjust unsupported parameters (e.g., keep_alive)
        self._is_cloud = isinstance(host, str) and ("ollama.com" in host)
        
        # Initialize Ollama client with proper authentication
        # Note: Ollama Turbo uses Authorization header without 'Bearer' prefix
        self._client = Client(
            host=host,
            headers={'Authorization': api_key}
        )
        
        # Ensure headers mapping is accessible for idempotency key management.
        # The upstream ollama Client doesn't expose .headers publicly, so we bind it
        # to the underlying httpx client's headers when available.
        try:
            if not hasattr(self._client, 'headers') and hasattr(self._client, '_client'):
                self._client.headers = self._client._client.headers  # type: ignore[attr-defined]
        except Exception:
            pass
        
        # Wrap with retry logic
        self._retry_client = RetryableOllamaClient(self._client, logger=self._logger)
        
        # Generation settings
        # Keep-alive must be a string per server expectations (e.g., '-1', '5m').
        # Read from env; default to '-1' (keep warmed indefinitely) unless overridden.
        self._keep_alive: Optional[str] = os.getenv('OLLAMA_KEEP_ALIVE', '-1')
        if self._keep_alive is not None:
            self._keep_alive = str(self._keep_alive).strip()
            if self._keep_alive == '' or self._keep_alive.lower() in {'false', '0', 'no'}:
                self._keep_alive = None
            else:
                # Coerce bare numeric values to seconds (e.g., '1' -> '1s').
                # Preserve special indefinite value '-1' and any values that already have units.
                try:
                    if self._keep_alive.isdigit():
                        self._keep_alive = f"{self._keep_alive}s"
                except Exception:
                    pass
        
        self._logger.info(f"Ollama adapter initialized - Model: {model}, Host: {host}")
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        context: ConversationContext,
        tools: Optional[List[ToolSchema]] = None
    ) -> ConversationResult:
        """Send chat request to Ollama Turbo."""
        try:
            # Prepare request parameters
            kwargs = {
                'model': self._model,
                'messages': messages,
            }
            _opts = context.to_generation_options()
            if _opts:
                kwargs['options'] = _opts
            
            # Add keep-alive for server-side model warming (local servers only)
            # Turbo cloud (ollama.com) does not document or require keep_alive; omit it there.
            if self._keep_alive and not self._is_cloud:
                kwargs['keep_alive'] = self._keep_alive
            
            # Add tools if provided and enabled
            if tools and context.enable_tools:
                tool_schemas: List[Dict[str, Any]] = []
                for t in tools:
                    if isinstance(t, dict):
                        tool_schemas.append(t)
                    else:
                        try:
                            tool_schemas.append(t.to_api_format())  # type: ignore[attr-defined]
                        except Exception:
                            continue
                kwargs['tools'] = tool_schemas
            
            # Generate idempotency key and attach to headers for this request
            idempotency_key = str(uuid.uuid4())
            try:
                self.set_idempotency_key(idempotency_key)
            except Exception:
                pass
            
            try:
                # Execute request with retry logic
                response = self._retry_client.chat(**kwargs)
            finally:
                # Clear header to avoid leaking across unrelated requests
                try:
                    self.clear_idempotency_key()
                except Exception:
                    pass

            # Extract content from response
            content = ""
            raw_message: Optional[Dict[str, Any]] = None
            if hasattr(response, 'message') and response.message:
                # SDK-style object
                raw_message = response.message
                content = raw_message.get('content', '')
            elif isinstance(response, dict):
                # Dict response
                raw_message = response.get('message')
                if raw_message is not None:
                    content = raw_message.get('content', '')
                else:
                    content = response.get('content', str(response))
            else:
                content = str(response)

            # Normalize tool_calls to simplified Harmony shape if present
            simplified_calls = None
            try:
                if isinstance(raw_message, dict) and raw_message.get('tool_calls'):
                    simplified_calls = []
                    for call in raw_message.get('tool_calls', []) or []:
                        if isinstance(call, dict):
                            if 'function' in call:
                                fn = call.get('function') or {}
                                name = fn.get('name')
                                args_raw = fn.get('arguments', '{}')
                                try:
                                    args_obj = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                                except Exception:
                                    args_obj = {}
                                simplified_calls.append({'name': name, 'arguments': args_obj, 'id': call.get('id')})
                            elif 'name' in call and 'arguments' in call:
                                simplified_calls.append({'name': call.get('name'), 'arguments': call.get('arguments') or {}, 'id': call.get('id')})
            except Exception:
                simplified_calls = None

            return ConversationResult(
                success=True,
                content=content,
                state=ConversationState.COMPLETED,
                metadata={
                    'model': self._model,
                    'idempotency_key': idempotency_key,
                    'tools_enabled': tools is not None,
                    # Surface raw message and tool_calls (simplified) for orchestrator parsing
                    'message': raw_message,
                    'tool_calls': simplified_calls,
                    'raw_response_type': type(response).__name__
                }
            )
            
        except Exception as e:
            # Log at DEBUG to avoid leaking errors to CLI output (legacy facade shares this logger)
            self._logger.debug(f"Ollama chat request failed: {e}")
            return ConversationResult(
                success=False,
                content=f"Ollama API error: {e}",
                state=ConversationState.ERROR,
                error=str(e)
            )
    
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        context: ConversationContext,
        tools: Optional[List[ToolSchema]] = None
    ) -> Iterator[str]:
        """Send streaming chat request to Ollama Turbo."""
        try:
            # Prepare request parameters
            kwargs = {
                'model': self._model,
                'messages': messages,
                'stream': True
            }
            _opts = context.to_generation_options()
            if _opts:
                kwargs['options'] = _opts
            
            # Add keep-alive for local servers only; omit for Turbo cloud
            if self._keep_alive and not self._is_cloud:
                kwargs['keep_alive'] = self._keep_alive
            
            # Add tools if provided and enabled
            if tools and context.enable_tools:
                tool_schemas: List[Dict[str, Any]] = []
                for t in tools:
                    if isinstance(t, dict):
                        tool_schemas.append(t)
                    else:
                        try:
                            tool_schemas.append(t.to_api_format())  # type: ignore[attr-defined]
                        except Exception:
                            continue
                kwargs['tools'] = tool_schemas
            
            # Generate idempotency key and attach to headers for this request
            idempotency_key = str(uuid.uuid4())
            try:
                self.set_idempotency_key(idempotency_key)
            except Exception:
                pass

            # Execute streaming request with retry logic
            stream = None
            try:
                stream = self._retry_client.chat_stream(**kwargs)
            finally:
                # Note: header stays set during iteration; it will be cleared after streaming completes below
                pass
            
            # Process stream chunks (emit only deltas to avoid duplication)
            accumulated_text = ""
            try:
                for chunk in stream:
                    try:
                        # Extract content from chunk
                        content = ''
                        msg_dict: Optional[Dict[str, Any]] = None
                        if hasattr(chunk, 'message') and chunk.message:
                            msg_dict = chunk.message
                            content = msg_dict.get('content', '')
                        elif isinstance(chunk, dict):
                            msg_dict = chunk.get('message') if 'message' in chunk else None
                            if msg_dict is not None:
                                content = msg_dict.get('content', '')
                            else:
                                content = chunk.get('content', '')
                        else:
                            content = str(chunk)

                        if content:
                            delta = content
                            try:
                                if not accumulated_text:
                                    # First emission
                                    delta = content
                                    accumulated_text = content
                                else:
                                    if content.startswith(accumulated_text):
                                        # Provider sent cumulative content
                                        delta = content[len(accumulated_text):]
                                        accumulated_text = content
                                    elif len(content) < len(accumulated_text):
                                        # Provider sent token delta (shorter payload)
                                        delta = content
                                        accumulated_text += content
                                    elif accumulated_text in content:
                                        # Cumulative but with modified prefix
                                        idx = content.find(accumulated_text)
                                        delta = content[idx + len(accumulated_text):]
                                        accumulated_text = content
                                    else:
                                        # Fallback: avoid echoing identical text
                                        delta = content if content != accumulated_text else ''
                                        if len(content) > len(accumulated_text):
                                            accumulated_text = content
                                        else:
                                            accumulated_text += delta
                            except Exception:
                                # On any issue, emit the raw content but advance accumulator defensively
                                delta = content
                                try:
                                    accumulated_text += delta
                                except Exception:
                                    pass

                            if delta:
                                yield delta

                        # Emit tool call sentinel for stream processor if present
                        try:
                            tool_calls = None
                            if msg_dict and isinstance(msg_dict, dict):
                                tool_calls = msg_dict.get('tool_calls')
                            if tool_calls:
                                for call in tool_calls:
                                    payload = {}
                                    if isinstance(call, dict) and 'function' in call:
                                        fn = call.get('function') or {}
                                        name = fn.get('name')
                                        args_raw = fn.get('arguments', '{}')
                                        try:
                                            args_obj = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                                        except Exception:
                                            args_obj = {}
                                        payload = {'name': name, 'arguments': args_obj, 'id': call.get('id')}
                                    elif isinstance(call, dict) and 'name' in call and 'arguments' in call:
                                        payload = {'name': call.get('name'), 'arguments': call.get('arguments') or {}, 'id': call.get('id')}
                                    yield f"<tool_call>{json.dumps(payload)}</tool_call>"
                        except Exception as ie:
                            # Never break the stream on tool-call extraction errors
                            self._logger.debug(f"Ignoring stream tool-call parse issue: {ie}")
                    except Exception as e:
                        self._logger.warning(f"Error processing stream chunk: {e}")
                        continue
            finally:
                try:
                    self.clear_idempotency_key()
                except Exception:
                    pass
                    
        except Exception as e:
            # Do not emit error chunks in the stream; allow caller to handle silent fallback
            self._logger.debug(f"Ollama streaming request failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            # Simple availability check
            test_messages = [{"role": "user", "content": "test"}]
            test_context = ConversationContext(
                model=self._model,
                max_output_tokens=1
            )
            
            result = self.chat(test_messages, test_context)
            return result.success
            
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self._model,
            "provider": "ollama_turbo",
            "host": self._host,
            "streaming_supported": True,
            "tool_calling_supported": True,
            "max_context_tokens": 120000,  # gpt-oss:120b context window
            "description": "OpenAI GPT-OSS 120B model via Ollama Turbo cloud service"
        }
    
    def set_idempotency_key(self, key: str) -> None:
        """Set idempotency key for requests (for retry consistency)."""
        # Add to headers for future requests
        if hasattr(self._client, 'headers'):
            self._client.headers['Idempotency-Key'] = key
    
    def clear_idempotency_key(self) -> None:
        """Clear idempotency key from headers."""
        if hasattr(self._client, 'headers') and 'Idempotency-Key' in self._client.headers:
            del self._client.headers['Idempotency-Key']
