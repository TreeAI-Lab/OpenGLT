"""Strict, opt-in efficiency benchmarking helpers for OpenGLT."""

from .runtime import benchmark_inference, efficiency_enabled, write_result

__all__ = ["benchmark_inference", "efficiency_enabled", "write_result"]
