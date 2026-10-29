import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters, stdio_client
from mcp.types import TextContent

_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).with_name("mcp_server.py"))],
)


async def _call(name: str, arguments: dict) -> str:
    async with Client(stdio_client(_PARAMS)) as client:
        result = await client.call_tool(name, arguments)
    text = ""
    for block in result.content:
        if isinstance(block, TextContent):
            text += block.text
    return text


def call_mcp_tool(name: str, arguments: dict) -> str:
    """同步包裝：讓 app.py 這種一般（非 async）的程式也能呼叫 MCP 工具"""
    return asyncio.run(_call(name, arguments))