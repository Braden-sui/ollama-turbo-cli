"""
Mem0 client adapter implementation.
Infrastructure layer adapter implementing the MemoryStore protocol.
"""

from __future__ import annotations
import logging
import os
import asyncio
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from ...domain.models.memory import MemoryEntry, MemoryContext, MemoryOperationResult, MemoryOperationType
from ...domain.interfaces.memory_store import MemoryStore

# Support both Mem0 platform and open-source SDKs.
# Prefer MemoryClient if available; otherwise fall back to Memory.
try:
    from mem0 import MemoryClient as _Mem0Client  # type: ignore
except ImportError:
    try:
        from mem0 import Memory as _Mem0Client  # type: ignore
    except ImportError:
        _Mem0Client = None  # type: ignore

# Backward-compatible alias used throughout this adapter and in tests
MemoryClient = _Mem0Client  # type: ignore


class Mem0Adapter(MemoryStore):
    """Adapter for Mem0 memory service."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._api_key = api_key or os.getenv('MEM0_API_KEY')
        self._user_id = user_id or os.getenv('MEM0_USER_ID', 'default-user')
        self._client: Optional[MemoryClient] = None
        
        # Configuration from environment
        self._timeout_connect_ms = int(os.getenv('MEM0_TIMEOUT_CONNECT_MS', '1000'))
        self._timeout_read_ms = int(os.getenv('MEM0_TIMEOUT_READ_MS', '2000'))
        
        if MemoryClient:
            try:
                if self._api_key:
                    try:
                        # Platform SDKs typically accept api_key
                        self._client = MemoryClient(api_key=self._api_key)
                    except TypeError:
                        # Open-source SDK may not accept api_key in constructor
                        self._client = MemoryClient()  # type: ignore[call-arg]
                else:
                    # Allow construction without api_key for SDKs using env vars
                    self._client = MemoryClient()
                self._logger.debug("Mem0 client initialized successfully")
            except Exception as e:
                self._logger.error(f"Failed to initialize Mem0 client: {e}")
                self._client = None
    
    def is_available(self) -> bool:
        """Check if Mem0 service is available."""
        return self._client is not None and self._api_key is not None
    
    async def add_memory(
        self,
        messages: List[Dict[str, Any]],
        context: MemoryContext,
        infer: bool = True
    ) -> MemoryOperationResult:
        """Add new memory from messages."""
        if not self.is_available():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.ADD,
                error="Mem0 service not available"
            )
        
        try:
            # Prepare kwargs with fallback patterns for different SDK versions
            kwargs: Dict[str, Any] = {
                "user_id": context.user_id or self._user_id,
            }
            # Add optional context fields
            if context.agent_id:
                kwargs["agent_id"] = context.agent_id
            if context.run_id:
                kwargs["run_id"] = context.run_id
            if context.metadata:
                kwargs["metadata"] = context.metadata

            # Try preferred signature with explicit infer param
            try:
                result = await asyncio.to_thread(
                    self._client.add,  # type: ignore[union-attr]
                    messages,
                    infer=infer,
                    **kwargs
                )
            except TypeError:
                # Fallback: some SDKs may not accept 'infer'
                try:
                    result = await asyncio.to_thread(
                        self._client.add,  # type: ignore[union-attr]
                        messages,
                        **kwargs
                    )
                except TypeError:
                    # Minimal fallback: just messages and user_id
                    result = await asyncio.to_thread(
                        self._client.add,  # type: ignore[union-attr]
                        messages,
                        user_id=context.user_id or self._user_id
                    )
            
            return MemoryOperationResult(
                success=True,
                operation_type=MemoryOperationType.ADD,
                data=result,
                metadata={"method": "add"}
            )
            
        except Exception as e:
            self._logger.error(f"Failed to add memory: {e}")
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.ADD,
                error=str(e)
            )
    
    async def search_memories(
        self,
        query: str,
        context: MemoryContext,
        limit: Optional[int] = None
    ) -> MemoryOperationResult:
        """Search for memories matching query."""
        if not self.is_available():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.SEARCH,
                error="Mem0 service not available"
            )
        
        try:
            filters = context.to_filters()

            # Preferred signature: query + user_id (+ limit)
            kwargs1: Dict[str, Any] = {
                "query": query,
                "user_id": context.user_id or self._user_id,
            }
            if limit:
                kwargs1["limit"] = limit

            try:
                raw_results = await asyncio.to_thread(
                    self._client.search,  # type: ignore[union-attr]
                    **kwargs1
                )
            except TypeError:
                # Fallback: some SDKs may expect filters instead of user_id
                kwargs2: Dict[str, Any] = {"query": query, "filters": filters}
                if limit:
                    kwargs2["limit"] = limit
                raw_results = await asyncio.to_thread(
                    self._client.search,  # type: ignore[union-attr]
                    **kwargs2
                )
            
            # Convert to MemoryEntry objects
            entries = []
            for item in raw_results or []:
                try:
                    entry = MemoryEntry.from_mem0_response(item)
                    entries.append(entry)
                except Exception as e:
                    self._logger.warning(f"Failed to parse memory entry: {e}")
            
            return MemoryOperationResult(
                success=True,
                operation_type=MemoryOperationType.SEARCH,
                data=entries,
                metadata={"query": query, "count": len(entries)}
            )
            
        except Exception as e:
            self._logger.error(f"Failed to search memories: {e}")
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.SEARCH,
                error=str(e)
            )
    
    async def get_memory(self, memory_id: str) -> MemoryOperationResult:
        """Get a specific memory by ID."""
        if not self.is_available():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.SEARCH,
                error="Mem0 service not available"
            )
        
        try:
            try:
                result = await asyncio.to_thread(self._client.get, memory_id=memory_id)  # type: ignore[union-attr]
            except TypeError:
                try:
                    result = await asyncio.to_thread(self._client.get, id=memory_id)  # type: ignore[union-attr]
                except TypeError:
                    # Positional-only fallback
                    result = await asyncio.to_thread(self._client.get, memory_id)  # type: ignore[union-attr]
            entry = MemoryEntry.from_mem0_response(result)
            
            return MemoryOperationResult(
                success=True,
                operation_type=MemoryOperationType.SEARCH,
                data=entry,
                metadata={"memory_id": memory_id}
            )
            
        except Exception as e:
            self._logger.error(f"Failed to get memory {memory_id}: {e}")
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.SEARCH,
                error=str(e)
            )
    
    async def get_all_memories(
        self,
        context: MemoryContext,
        limit: Optional[int] = None
    ) -> MemoryOperationResult:
        """Get all memories for context."""
        if not self.is_available():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.GET_ALL,
                error="Mem0 service not available"
            )
        
        try:
            filters = context.to_filters()

            # Preferred signature: user_id (+ limit)
            kwargs1: Dict[str, Any] = {"user_id": context.user_id or self._user_id}
            if limit:
                kwargs1["limit"] = limit

            try:
                raw_results = await asyncio.to_thread(
                    self._client.get_all,  # type: ignore[union-attr]
                    **kwargs1
                )
            except TypeError:
                # Fallback to filters-based signature
                kwargs2: Dict[str, Any] = {"filters": filters}
                if limit:
                    kwargs2["limit"] = limit
                raw_results = await asyncio.to_thread(
                    self._client.get_all,  # type: ignore[union-attr]
                    **kwargs2
                )
            
            # Convert to MemoryEntry objects
            entries = []
            for item in raw_results or []:
                try:
                    entry = MemoryEntry.from_mem0_response(item)
                    entries.append(entry)
                except Exception as e:
                    self._logger.warning(f"Failed to parse memory entry: {e}")
            
            return MemoryOperationResult(
                success=True,
                operation_type=MemoryOperationType.GET_ALL,
                data=entries,
                metadata={"count": len(entries)}
            )
            
        except Exception as e:
            self._logger.error(f"Failed to get all memories: {e}")
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.GET_ALL,
                error=str(e)
            )
    
    async def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        data: Optional[str] = None
    ) -> MemoryOperationResult:
        """Update an existing memory."""
        if not self.is_available():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.UPDATE,
                error="Mem0 service not available"
            )
        
        try:
            # Try with 'text' parameter first
            kwargs = {"memory_id": memory_id}
            if text:
                kwargs["text"] = text
            elif data:
                kwargs["data"] = data
            else:
                return MemoryOperationResult(
                    success=False,
                    operation_type=MemoryOperationType.UPDATE,
                    error="No text or data provided for update"
                )
            
            try:
                result = await asyncio.to_thread(self._client.update, **kwargs)
            except TypeError:
                # Fallback to 'data' parameter if 'text' not supported
                kwargs = {"memory_id": memory_id, "data": text or data}
                result = await asyncio.to_thread(self._client.update, **kwargs)
            
            return MemoryOperationResult(
                success=True,
                operation_type=MemoryOperationType.UPDATE,
                data=result,
                metadata={"memory_id": memory_id}
            )
            
        except Exception as e:
            self._logger.error(f"Failed to update memory {memory_id}: {e}")
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.UPDATE,
                error=str(e)
            )
    
    async def delete_memory(self, memory_id: str) -> MemoryOperationResult:
        """Delete a specific memory."""
        if not self.is_available():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.DELETE,
                error="Mem0 service not available"
            )
        
        try:
            result = await asyncio.to_thread(
                self._client.delete,
                memory_id=memory_id
            )
            
            return MemoryOperationResult(
                success=True,
                operation_type=MemoryOperationType.DELETE,
                data=result,
                metadata={"memory_id": memory_id}
            )
            
        except Exception as e:
            self._logger.error(f"Failed to delete memory {memory_id}: {e}")
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.DELETE,
                error=str(e)
            )
    
    async def delete_all_memories(self, context: MemoryContext) -> MemoryOperationResult:
        """Delete all memories for context."""
        if not self.is_available():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.DELETE,
                error="Mem0 service not available"
            )
        
        try:
            # Note: Mem0 delete_all typically requires user_id
            result = await asyncio.to_thread(
                self._client.delete_all,
                user_id=context.user_id or self._user_id
            )
            
            return MemoryOperationResult(
                success=True,
                operation_type=MemoryOperationType.DELETE,
                data=result,
                metadata={"user_id": context.user_id or self._user_id}
            )
            
        except Exception as e:
            self._logger.error(f"Failed to delete all memories: {e}")
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.DELETE,
                error=str(e)
            )
    
    async def link_memories(
        self,
        memory1_id: str,
        memory2_id: str,
        context: MemoryContext
    ) -> MemoryOperationResult:
        """Link two memories together."""
        if not self.is_available():
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.LINK,
                error="Mem0 service not available"
            )
        
        try:
            # Note: Linking may not be available in all plans/SDK versions
            try:
                result = await asyncio.to_thread(
                    self._client.link,  # type: ignore[union-attr]
                    memory1_id=memory1_id,
                    memory2_id=memory2_id,
                    user_id=context.user_id or self._user_id
                )
            except TypeError:
                # Positional fallback
                result = await asyncio.to_thread(
                    self._client.link,  # type: ignore[union-attr]
                    memory1_id,
                    memory2_id,
                    context.user_id or self._user_id
                )
            
            return MemoryOperationResult(
                success=True,
                operation_type=MemoryOperationType.LINK,
                data=result,
                metadata={"memory1_id": memory1_id, "memory2_id": memory2_id}
            )
            
        except Exception as e:
            self._logger.error(f"Failed to link memories: {e}")
            return MemoryOperationResult(
                success=False,
                operation_type=MemoryOperationType.LINK,
                error=str(e)
            )
