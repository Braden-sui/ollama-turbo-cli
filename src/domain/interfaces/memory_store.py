"""
Memory store protocol interface.
Defines the contract for memory storage implementations.
"""

from __future__ import annotations
from typing import Protocol, List, Optional, Dict, Any
from ..models.memory import MemoryEntry, MemoryContext, MemoryOperation, MemoryOperationResult


class MemoryStore(Protocol):
    """Protocol for memory storage implementations."""
    
    async def add_memory(
        self,
        messages: List[Dict[str, Any]],
        context: MemoryContext,
        infer: bool = True
    ) -> MemoryOperationResult:
        """Add new memory from messages."""
        ...
    
    async def search_memories(
        self,
        query: str,
        context: MemoryContext,
        limit: Optional[int] = None
    ) -> MemoryOperationResult:
        """Search for memories matching query."""
        ...
    
    async def get_memory(self, memory_id: str) -> MemoryOperationResult:
        """Get a specific memory by ID."""
        ...
    
    async def get_all_memories(
        self,
        context: MemoryContext,
        limit: Optional[int] = None
    ) -> MemoryOperationResult:
        """Get all memories for context."""
        ...
    
    async def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        data: Optional[str] = None
    ) -> MemoryOperationResult:
        """Update an existing memory."""
        ...
    
    async def delete_memory(self, memory_id: str) -> MemoryOperationResult:
        """Delete a specific memory."""
        ...
    
    async def delete_all_memories(self, context: MemoryContext) -> MemoryOperationResult:
        """Delete all memories for context."""
        ...
    
    async def link_memories(
        self,
        memory1_id: str,
        memory2_id: str,
        context: MemoryContext
    ) -> MemoryOperationResult:
        """Link two memories together."""
        ...
    
    def is_available(self) -> bool:
        """Check if memory store is available and configured."""
        ...


class MemoryCircuitBreaker(Protocol):
    """Protocol for circuit breaker implementations."""
    
    def is_request_allowed(self) -> bool:
        """Check if request should be allowed through circuit breaker."""
        ...
    
    def record_success(self) -> None:
        """Record successful operation."""
        ...
    
    def record_failure(self, error: Exception) -> None:
        """Record failed operation."""
        ...
    
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        ...


class BackgroundWorker(Protocol):
    """Protocol for background task processing."""
    
    def enqueue_task(self, operation: MemoryOperation) -> bool:
        """Enqueue a memory operation for background processing."""
        ...
    
    def start(self) -> None:
        """Start the background worker."""
        ...
    
    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the background worker."""
        ...
    
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        ...
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        ...
