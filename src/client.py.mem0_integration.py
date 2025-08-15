"""
Mem0 integration methods for OllamaTurboClient.
This shows the exact methods that replace the monolithic Mem0 implementation.
"""

import asyncio
import os
import logging
from typing import Optional, List, Dict, Any

def _init_mem0(self) -> None:
    """Initialize the new hexagonal memory architecture (replaces old _init_mem0)."""
    try:
        # Get configuration from environment
        api_key = os.getenv('MEM0_API_KEY')
        user_id = os.getenv('MEM0_USER_ID', 'default-user')
        
        if not api_key:
            self.mem0_enabled = False
            self._memory_service = None
            self._memory_context = None
            return
        
        # Create memory context
        self._memory_context = MemoryContext(
            user_id=user_id,
            agent_id="ollama-turbo-cli",
            metadata={"source": "cli", "version": "1.0"}
        )
        
        # Create infrastructure components
        memory_store = Mem0Adapter(api_key=api_key, user_id=user_id, logger=self.logger)
        background_worker = Mem0BackgroundWorker(memory_store, logger=self.logger)
        circuit_breaker = Mem0CircuitBreaker(logger=self.logger)
        
        # Create application service
        self._memory_service = MemoryService(
            memory_store=memory_store,
            background_worker=background_worker,
            circuit_breaker=circuit_breaker,
            logger=self.logger
        )
        
        # Start background worker
        if self._memory_service._worker:
            self._memory_service._worker.start()
        
        self.mem0_enabled = self._memory_service.is_enabled()
        self.mem0_user_id = user_id  # For backward compatibility
        
        if self.mem0_enabled:
            self.logger.info(f"Mem0 memory service initialized (user: {user_id})")
        else:
            self.logger.warning("Mem0 memory service not available")
            
    except Exception as e:
        self.logger.error(f"Failed to initialize Mem0: {e}")
        self.mem0_enabled = False
        self._memory_service = None
        self._memory_context = None


def _inject_mem0_context(self, user_message: str) -> None:
    """Inject relevant memories into conversation context (replaces old method)."""
    if not self._memory_service or not self._memory_context:
        return
    
    try:
        # Use asyncio to run the async memory search
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            self._memory_service.search_memories(
                query=user_message,
                context=self._memory_context,
                limit=3
            )
        )
        
        if result.success and result.entries:
            # Create memory context injection
            memory_texts = []
            for entry in result.entries:
                memory_texts.append(f"- {entry.memory}")
            
            context_injection = (
                "Relevant memories from our previous conversations:\n" + 
                "\n".join(memory_texts) + 
                "\n\nPlease consider this context when responding.\n"
            )
            
            # Add as system message
            self.conversation_history.append({
                'role': 'system',
                'content': context_injection
            })
        
        loop.close()
        
    except Exception as e:
        self.logger.debug(f"Failed to inject memory context: {e}")


def _handle_mem0_nlu_command(self, text: str) -> bool:
    """Handle natural language memory commands (replaces old method)."""
    if not self._memory_service or not self._memory_context:
        return False
    
    try:
        # Use asyncio to run the async command processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            self._memory_service.process_natural_language_command(
                text=text,
                context=self._memory_context
            )
        )
        
        if result:
            if result.success:
                # Handle different command types
                if result.operation_type.value == "add":
                    print("✅ Remembered.")
                elif result.operation_type.value == "search":
                    if result.entries:
                        print("🧠 Here's what I recall:")
                        for i, entry in enumerate(result.entries[:5], 1):
                            print(f"  {i}. {entry.memory}")
                    else:
                        print("🤔 I don't have anything saved yet.")
                elif result.operation_type.value == "get_all":
                    if result.entries:
                        print("🧠 Memories:")
                        for i, entry in enumerate(result.entries[:10], 1):
                            print(f"  {i}. {entry.memory}")
                    else:
                        print("📭 No memories found.")
                elif result.operation_type.value == "delete":
                    deleted_count = result.metadata.get("deleted_count", 0)
                    print("🗑️ Forgotten." if deleted_count > 0 else "ℹ️ Nothing to forget matched.")
                elif result.operation_type.value == "update":
                    print("✅ Updated.")
            else:
                print(f"❌ Memory operation failed: {result.error}")
            
            loop.close()
            return True
        
        loop.close()
        return False
        
    except Exception as e:
        self.logger.debug(f"Memory NLU command error: {e}")
        return False


def _mem0_background_add(self, user_msg: str, assistant_msg: str) -> None:
    """Add conversation to memory in background (replaces old method)."""
    if not self._memory_service or not self._memory_context:
        return
    
    try:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
        
        # Use asyncio to run the async add operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(
            self._memory_service.add_memory(
                messages=messages,
                context=self._memory_context,
                background=True
            )
        )
        
        loop.close()
        
    except Exception as e:
        self.logger.debug(f"Background memory add failed: {e}")


def _shutdown_mem0(self) -> None:
    """Graceful shutdown of memory service (new method)."""
    if self._memory_service and hasattr(self._memory_service, '_worker'):
        try:
            if self._memory_service._worker:
                self._memory_service._worker.stop(timeout=5.0)
            self.logger.debug("Memory service shut down gracefully")
        except Exception as e:
            self.logger.error(f"Error during memory service shutdown: {e}")
