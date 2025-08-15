"""
Background worker for asynchronous memory operations.
Handles queued memory tasks with retry logic and graceful shutdown.
"""

from __future__ import annotations
import logging
import os
import queue
import threading
import time
from typing import Optional
from datetime import datetime

from ...domain.models.memory import MemoryOperation, BackgroundTask
from ...domain.interfaces.memory_store import BackgroundWorker, MemoryStore


class Mem0BackgroundWorker(BackgroundWorker):
    """Background worker for processing memory operations asynchronously."""
    
    def __init__(
        self,
        memory_store: MemoryStore,
        max_queue_size: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        self._store = memory_store
        self._logger = logger or logging.getLogger(__name__)
        
        # Configuration from environment
        self._max_queue_size = max_queue_size or int(os.getenv('MEM0_ADD_QUEUE_MAX', '256'))
        
        # Worker state
        self._queue: queue.Queue[BackgroundTask] = queue.Queue(maxsize=self._max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        # Statistics
        self._processed_count = 0
        self._failed_count = 0
        self._last_activity = datetime.now()
    
    def enqueue_task(self, operation: MemoryOperation) -> bool:
        """Enqueue a memory operation for background processing."""
        if not self._running:
            self._logger.warning("Background worker not running, cannot enqueue task")
            return False
        
        try:
            task = BackgroundTask(operation=operation)
            self._queue.put_nowait(task)
            self._logger.debug(f"Enqueued {operation.operation_type.value} task")
            return True
        except queue.Full:
            self._logger.warning(f"Memory queue full (size: {self._max_queue_size}), dropping task")
            return False
    
    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            self._logger.warning("Background worker already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="Mem0BackgroundWorker",
            daemon=True
        )
        self._worker_thread.start()
        self._logger.info("Memory background worker started")
    
    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the background worker thread."""
        if not self._running:
            return
        
        self._logger.info("Stopping memory background worker...")
        self._stop_event.set()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            
            if self._worker_thread.is_alive():
                self._logger.warning("Background worker did not stop gracefully within timeout")
            else:
                self._logger.info("Background worker stopped gracefully")
        
        self._running = False
        
        # Log final statistics
        self._logger.info(
            f"Worker statistics - Processed: {self._processed_count}, "
            f"Failed: {self._failed_count}, Queue size: {self.get_queue_size()}"
        )
    
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self._running and (
            self._worker_thread is not None and self._worker_thread.is_alive()
        )
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def get_statistics(self) -> dict:
        """Get worker statistics."""
        return {
            "running": self.is_running(),
            "queue_size": self.get_queue_size(),
            "processed_count": self._processed_count,
            "failed_count": self._failed_count,
            "last_activity": self._last_activity.isoformat(),
            "max_queue_size": self._max_queue_size
        }
    
    def _worker_loop(self) -> None:
        """Main worker loop processing queued tasks."""
        self._logger.debug("Background worker loop started")
        
        while not self._stop_event.is_set():
            try:
                # Wait for tasks with timeout to allow periodic stop checks
                try:
                    task = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self._last_activity = datetime.now()
                
                # Process the task
                success = self._process_task(task)
                
                if success:
                    self._processed_count += 1
                    self._logger.debug(f"Successfully processed {task.operation.operation_type.value} task")
                else:
                    self._failed_count += 1
                    
                    # Retry logic for failed tasks
                    if task.should_retry():
                        task.increment_attempts()
                        self._logger.debug(f"Retrying task (attempt {task.attempts}/{task.max_attempts})")
                        
                        # Add delay before retry
                        time.sleep(min(2 ** task.attempts, 30))  # Exponential backoff, max 30s
                        
                        try:
                            self._queue.put_nowait(task)
                        except queue.Full:
                            self._logger.warning("Queue full, dropping retry task")
                    else:
                        self._logger.error(
                            f"Task failed permanently after {task.attempts} attempts: "
                            f"{task.operation.operation_type.value}"
                        )
                
                self._queue.task_done()
                
            except Exception as e:
                self._logger.error(f"Unexpected error in worker loop: {e}")
                time.sleep(1)  # Brief pause to prevent tight error loops
        
        self._logger.debug("Background worker loop finished")
    
    def _process_task(self, task: BackgroundTask) -> bool:
        """Process a single background task."""
        try:
            operation = task.operation
            
            # Import asyncio here to avoid issues in thread
            import asyncio
            
            # Create new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Execute the operation
            if operation.operation_type.value == "add":
                result = loop.run_until_complete(
                    self._store.add_memory(
                        operation.messages or [],
                        operation.context,
                        operation.data.get("infer", True)
                    )
                )
            else:
                self._logger.warning(f"Unsupported background operation: {operation.operation_type.value}")
                return False
            
            if not result.success:
                self._logger.warning(f"Memory operation failed: {result.error}")
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to process background task: {e}")
            return False
    
    def __del__(self):
        """Cleanup when worker is destroyed."""
        if self._running:
            self.stop(timeout=5.0)
