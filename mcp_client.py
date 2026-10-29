import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).with_name("mcp_server.py"))],
)


async def _call(name: str, arguments: dict) -> str:
    async with stdio_client(_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=arguments)
    text = ""
    for block in result.content:
        if isinstance(block, TextContent):
            text += block.text
    return text


def call_mcp_tool(name: str, arguments: dict) -> str:
    """同步包裝：讓 app.py 這種一般（非 async）的程式也能呼叫 MCP 工具"""
    return asyncio.run(_call(name, arguments))

async def _call_many(calls: list) -> list:
    async with stdio_client(_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            texts = []
            for name, arguments in calls:
                result = await session.call_tool(name, arguments=arguments)
                text = ""
                for block in result.content:
                    if isinstance(block, TextContent):
                        text += block.text
                texts.append(text)
    return texts


def call_mcp_tools(calls: list) -> list:
    """一次連線處理多個工具呼叫。calls 是 [(工具名稱, 參數字典), ...]，回傳結果文字清單"""
    return asyncio.run(_call_many(calls))