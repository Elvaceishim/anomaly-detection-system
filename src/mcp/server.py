"""
MCP Server for Anomaly Detection System.

This server exposes controlled tools for LLM-assisted fraud review.
It integrates with the main FastAPI application to access transaction
data and model state.

Usage:
    # As a standalone server
    python -m src.mcp.server
    
    # Or integrate with the FastAPI app
    from src.mcp.server import create_mcp_server
"""

import asyncio
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .tools import MCPTools
from .schemas import (
    TransactionIdInput,
    UserIdInput,
    HumanDecisionInput
)


def create_mcp_server(transaction_store=None, model_state=None) -> Server:
    """
    Create and configure the MCP server with all tools.
    
    Args:
        transaction_store: The API's transaction store for data access
        model_state: The loaded model state for explanations
    
    Returns:
        Configured MCP Server instance
    """
    server = Server("anomaly-detection-mcp")
    tools = MCPTools(transaction_store=transaction_store, model_state=model_state)
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available MCP tools."""
        return [
            Tool(
                name="get_transaction_summary",
                description=(
                    "Get a safe summary of a transaction including amount, merchant, "
                    "timestamp, location, and risk score. Does NOT include PII, "
                    "fraud outcomes, or sensitive card details."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": "The unique transaction identifier"
                        }
                    },
                    "required": ["transaction_id"]
                }
            ),
            Tool(
                name="get_user_behavior_snapshot",
                description=(
                    "Get aggregated behavioral statistics for a user including "
                    "transaction count, average amount, velocity, and merchant diversity. "
                    "Does NOT include full transaction history or exact timestamps."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The unique user identifier"
                        }
                    },
                    "required": ["user_id"]
                }
            ),
            Tool(
                name="get_anomaly_signals",
                description=(
                    "Get anomaly signals for a transaction including amount percentile, "
                    "z-score, velocity spikes, and location/merchant changes. "
                    "Helps understand why a transaction might be suspicious."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": "The transaction to analyze"
                        }
                    },
                    "required": ["transaction_id"]
                }
            ),
            Tool(
                name="get_model_explanation",
                description=(
                    "Get an explanation of why the model assigned a specific risk score. "
                    "Returns top contributing features and a human-readable summary. "
                    "Does NOT expose model weights or exact thresholds."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": "The transaction to explain"
                        }
                    },
                    "required": ["transaction_id"]
                }
            ),
            Tool(
                name="log_human_decision",
                description=(
                    "Log a human review decision (approve/reject/escalate) for audit. "
                    "This is the ONLY write operation available. All other tools are read-only."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": "The transaction being reviewed"
                        },
                        "decision": {
                            "type": "string",
                            "enum": ["approve", "reject", "escalate"],
                            "description": "The analyst's decision"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional notes explaining the decision"
                        },
                        "analyst_id": {
                            "type": "string",
                            "description": "Optional ID of the analyst"
                        }
                    },
                    "required": ["transaction_id", "decision"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute a tool and return results."""
        
        if name == "get_transaction_summary":
            result = tools.get_transaction_summary(arguments["transaction_id"])
        
        elif name == "get_user_behavior_snapshot":
            result = tools.get_user_behavior_snapshot(arguments["user_id"])
        
        elif name == "get_anomaly_signals":
            result = tools.get_anomaly_signals(arguments["transaction_id"])
        
        elif name == "get_model_explanation":
            result = tools.get_model_explanation(arguments["transaction_id"])
        
        elif name == "log_human_decision":
            result = tools.log_human_decision(
                transaction_id=arguments["transaction_id"],
                decision=arguments["decision"],
                notes=arguments.get("notes"),
                analyst_id=arguments.get("analyst_id")
            )
        
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    return server


async def main():
    """Run the MCP server in stdio mode."""
    server = create_mcp_server()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
