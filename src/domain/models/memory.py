"""
Domain models for memory management.
Pure business logic with no external dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


class MemoryOperationType(Enum):
    """Types of memory operations."""
    ADD = "add"
    SEARCH = "search"
    UPDATE = "update"
    DELETE = "delete"
    GET_ALL = "get_all"
    LINK = "link"


@dataclass(frozen=True)
class MemoryContext:
    """Context information for memory operations."""
    user_id: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_filters(self) -> Dict[str, Any]:
        """Convert to filters dictionary for memory client."""
        filters = {"user_id": self.user_id}
        if self.agent_id:
            filters["agent_id"] = self.agent_id
        if self.run_id:
            filters["run_id"] = self.run_id
        return filters


@dataclass(frozen=True)
class MemoryEntry:
    """Represents a single memory entry."""
    id: str
    memory: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None  # For search results
    
    @classmethod
    def from_mem0_response(cls, data: Dict[str, Any]) -> MemoryEntry:
        """Create MemoryEntry from Mem0 API response."""
        memory_text = data.get('memory') or (data.get('data') or {}).get('memory') or ''
        
        return cls(
            id=data.get('id', ''),
            memory=memory_text,
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
            metadata=data.get('metadata', {}),
            score=data.get('score')
        )


@dataclass
class MemoryOperation:
    """Represents a memory operation request."""
    operation_type: MemoryOperationType
    context: MemoryContext
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Operation-specific fields
    query: Optional[str] = None
    memory_id: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    text: Optional[str] = None
    memory_ids: Optional[List[str]] = None  # For linking operations


@dataclass
class MemoryOperationResult:
    """Result of a memory operation."""
    success: bool
    operation_type: MemoryOperationType
    data: Optional[Union[List[MemoryEntry], MemoryEntry, Dict[str, Any]]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def entries(self) -> List[MemoryEntry]:
        """Get memory entries from result data."""
        if not self.data:
            return []
        if isinstance(self.data, list):
            return [entry for entry in self.data if isinstance(entry, MemoryEntry)]
        if isinstance(self.data, MemoryEntry):
            return [self.data]
        return []


@dataclass
class CircuitBreakerState:
    """State of the circuit breaker for memory operations."""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0
    
    def should_allow_request(self, threshold: int, cooldown_ms: int) -> bool:
        """Determine if request should be allowed based on circuit breaker state."""
        if not self.is_open:
            return True
            
        if self.last_failure_time is None:
            return True
            
        cooldown_elapsed = (datetime.now() - self.last_failure_time).total_seconds() * 1000 >= cooldown_ms
        return cooldown_elapsed
    
    def record_success(self) -> None:
        """Record a successful operation."""
        self.failure_count = 0
        self.success_count += 1
        if self.is_open:
            self.is_open = False
    
    def record_failure(self, threshold: int) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= threshold:
            self.is_open = True


@dataclass
class BackgroundTask:
    """Represents a background memory task."""
    operation: MemoryOperation
    created_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    
    def should_retry(self) -> bool:
        """Determine if task should be retried."""
        return self.attempts < self.max_attempts
    
    def increment_attempts(self) -> None:
        """Increment attempt counter."""
        self.attempts += 1
