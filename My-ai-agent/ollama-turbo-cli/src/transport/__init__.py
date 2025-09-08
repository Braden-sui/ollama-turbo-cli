"""
Transport layer abstractions: networking helpers and lightweight client wrapper.

This package exposes:
- networking: host resolution, keep-alive, idempotency header management
- policy: retry/backoff policy with retryable classification
- http: TransportHttpClient wrapper implementing policy around SDK client
- ollama_client: placeholder wrapper for explicit timeout control
"""

__all__ = ["networking", "ollama_client", "policy", "http"]
