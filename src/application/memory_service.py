"""
Memory service - Application layer orchestrating memory operations.
Coordinates between domain logic and infrastructure adapters.
"""

from __future__ import annotations
import logging
import asyncio
from typing import List, Optional, Dict, Any, Union
from dataclasses import asdict

from ..domain.models.memory import (
    MemoryEntry, MemoryContext, MemoryOperation, MemoryOperationResult,
    MemoryOperationType, BackgroundTask
)
from ..domain.interfaces.memory_store import MemoryStore, BackgroundWorker, MemoryCircuitBreaker


class MemoryService:
    """Application service for memory operations."""
    
    def __init__(
        self,
        memory_store: MemoryStore,
        background_worker: Optional[BackgroundWorker] = None,
        circuit_breaker: Optional[MemoryCircuitBreaker] = None,
        logger: Optional[logging.Logger] = None
    ):
        self._store = memory_store
        self._worker = background_worker
        self._circuit_breaker = circuit_breaker
        self._logger = logger or logging.getLogger(__name__)
    
    def is_enabled(self) -> bool:
        """Check if memory service is enabled and available."""
        return self._store.is_available()
    
    async def add_memory(
        self,
        messages: List[Dict[str, Any]],
        context: MemoryContext,
        infer: bool = True,
        background: bool = False
    ) -> MemoryOperationResult:
        """Add new memory from conversation messages."""
        if not self.is_enabled():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.ADD,
                error="Memory service not available"
            )
        
        operation = MemoryOperation(
            operation_type=MemoryOperationType.ADD,
            context=context,
            messages=messages,
            data={"infer": infer}
        )
        
        if background and self._worker:
            success = self._worker.enqueue_task(operation)
            return MemoryOperationResult(
                success=success,
                operation_type=MemoryOperationType.ADD,
                metadata={"queued": success}
            )
        
        return await self._execute_operation(operation)
    
    async def search_memories(
        self,
        query: str,
        context: MemoryContext,
        limit: Optional[int] = None
    ) -> MemoryOperationResult:
        """Search for relevant memories."""
        if not self.is_enabled():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.SEARCH,
                error="Memory service not available"
            )
        
        operation = MemoryOperation(
            operation_type=MemoryOperationType.SEARCH,
            context=context,
            query=query,
            data={"limit": limit} if limit else {}
        )
        
        return await self._execute_operation(operation)
    
    async def get_all_memories(
        self,
        context: MemoryContext,
        limit: Optional[int] = None
    ) -> MemoryOperationResult:
        """Get all memories for a context."""
        if not self.is_enabled():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.GET_ALL,
                error="Memory service not available"
            )
        
        operation = MemoryOperation(
            operation_type=MemoryOperationType.GET_ALL,
            context=context,
            data={"limit": limit} if limit else {}
        )
        
        return await self._execute_operation(operation)
    
    async def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        data: Optional[str] = None
    ) -> MemoryOperationResult:
        """Update an existing memory."""
        if not self.is_enabled():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.UPDATE,
                error="Memory service not available"
            )
        
        operation = MemoryOperation(
            operation_type=MemoryOperationType.UPDATE,
            context=MemoryContext(user_id=""),  # Will be populated by store if needed
            memory_id=memory_id,
            text=text,
            data={"data": data} if data else {}
        )
        
        return await self._execute_operation(operation)
    
    async def delete_memory(self, memory_id: str) -> MemoryOperationResult:
        """Delete a specific memory."""
        if not self.is_enabled():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.DELETE,
                error="Memory service not available"
            )
        
        operation = MemoryOperation(
            operation_type=MemoryOperationType.DELETE,
            context=MemoryContext(user_id=""),  # Will be populated by store if needed
            memory_id=memory_id
        )
        
        return await self._execute_operation(operation)
    
    async def link_memories(
        self,
        memory1_id: str,
        memory2_id: str,
        context: MemoryContext
    ) -> MemoryOperationResult:
        """Link two memories together."""
        if not self.is_enabled():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.LINK,
                error="Memory service not available"
            )
        
        operation = MemoryOperation(
            operation_type=MemoryOperationType.LINK,
            context=context,
            memory_ids=[memory1_id, memory2_id]
        )
        
        return await self._execute_operation(operation)
    
    async def process_natural_language_command(
        self,
        text: str,
        context: MemoryContext
    ) -> Optional[MemoryOperationResult]:
        """Process natural language memory commands."""
        text_lower = text.lower().strip()
        
        # Remember command
        if text_lower.startswith("remember "):
            content = text[9:].strip()
            if content:
                messages = [{"role": "user", "content": content}]
                return await self.add_memory(messages, context, background=True)
        
        # Forget command
        if text_lower.startswith("forget "):
            query = text[7:].strip()
            if query:
                search_result = await self.search_memories(query, context)
                if search_result.success and search_result.entries:
                    deleted_count = 0
                    for entry in search_result.entries:
                        if query.lower() in entry.memory.lower():
                            delete_result = await self.delete_memory(entry.id)
                            if delete_result.success:
                                deleted_count += 1
                    
                    return MemoryOperationResult(
                        success=True,
                        operation_type=MemoryOperationType.DELETE,
                        metadata={"deleted_count": deleted_count}
                    )
        
        # Update command
        if text_lower.startswith("update ") and " to " in text_lower:
            parts = text[7:].strip().split(" to ", 1)
            if len(parts) == 2:
                subject, new_text = parts[0].strip(), parts[1].strip()
                search_result = await self.search_memories(subject, context)
                if search_result.success and search_result.entries:
                    entry = search_result.entries[0]
                    return await self.update_memory(entry.id, text=new_text)
                else:
                    # Add as new memory if not found
                    messages = [{"role": "user", "content": new_text}]
                    return await self.add_memory(messages, context, background=True)
        
        # List command
        if any(cmd in text_lower for cmd in ["list memories", "show memories"]):
            return await self.get_all_memories(context, limit=10)
        
        # Search command
        if text_lower.startswith("search memories for "):
            query = text[20:].strip()
            if query:
                return await self.search_memories(query, context, limit=5)
        
        # Query patterns
        query_patterns = [
            "what do you know about me",
            "what did i tell you",
            "do you remember",
            "what do you remember about me"
        ]
        if any(pattern in text_lower for pattern in query_patterns):
            return await self.search_memories(text, context, limit=5)
        
        return None
    
    async def _execute_operation(self, operation: MemoryOperation) -> MemoryOperationResult:
        """Execute a memory operation with circuit breaker protection."""
        if self._circuit_breaker and not self._circuit_breaker.is_request_allowed():
            return MemoryOperationResult(
                success=False,
                operation_type=operation.operation_type,
                error="Circuit breaker is open"
            )
        
        try:
            result = await self._dispatch_operation(operation)
            
            if self._circuit_breaker:
                if result.success:
                    self._circuit_breaker.record_success()
                else:
                    self._circuit_breaker.record_failure(Exception(result.error or "Unknown error"))
            
            return result
            
        except Exception as e:
            self._logger.error(f"Memory operation failed: {e}")
            
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(e)
            
            return MemoryOperationResult(
                success=False,
                operation_type=operation.operation_type,
                error=str(e)
            )
    
    async def _dispatch_operation(self, operation: MemoryOperation) -> MemoryOperationResult:
        """Dispatch operation to appropriate store method."""
        if operation.operation_type == MemoryOperationType.ADD:
            return await self._store.add_memory(
                operation.messages or [],
                operation.context,
                operation.data.get("infer", True)
            )
        elif operation.operation_type == MemoryOperationType.SEARCH:
            return await self._store.search_memories(
                operation.query or "",
                operation.context,
                operation.data.get("limit")
            )
        elif operation.operation_type == MemoryOperationType.GET_ALL:
            return await self._store.get_all_memories(
                operation.context,
                operation.data.get("limit")
            )
        elif operation.operation_type == MemoryOperationType.UPDATE:
            return await self._store.update_memory(
                operation.memory_id or "",
                operation.text,
                operation.data.get("data")
            )
        elif operation.operation_type == MemoryOperationType.DELETE:
            return await self._store.delete_memory(operation.memory_id or "")
        elif operation.operation_type == MemoryOperationType.LINK:
            if operation.memory_ids and len(operation.memory_ids) >= 2:
                return await self._store.link_memories(
                    operation.memory_ids[0],
                    operation.memory_ids[1],
                    operation.context
                )
        
        return MemoryOperationResult(
            success=False,
            operation_type=operation.operation_type,
            error=f"Unknown operation type: {operation.operation_type}"
        )
