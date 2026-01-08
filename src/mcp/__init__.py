"""
MCP module for LLM-assisted fraud review.

This module provides a controlled interface between LLMs and the
anomaly detection system, exposing only approved read-only tools
and structured audit logging.
"""

from .server import create_mcp_server
from .tools import MCPTools

__all__ = ["create_mcp_server", "MCPTools"]
