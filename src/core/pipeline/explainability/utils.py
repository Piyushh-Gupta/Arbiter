"""Utility module for Pipeline Explainability (M5.5)."""

import hashlib


def generate_sha256_trace_id(data: str) -> str:
    """Generates a stable 16-character SHA-256 trace ID from input string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
