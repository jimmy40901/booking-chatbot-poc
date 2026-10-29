import asyncio
import sys
from pathlib import Path
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

def _params():
    # 子行程預設不會繼承我們的金鑰，這裡明確傳過去（只放記憶體，不寫檔、不印出）
    env = {}
    key = os.getenv("GEMINI_API_KEY")
    if key:
        env["GEMINI_API_KEY"] = key
    return StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("mcp_server.py"))],
        env=env or None,
    )


async def _call(name: str, arguments: dict) -> str:
    async with stdio_client(_params()) as (read, write):
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
    async with stdio_client(_params()) as (read, write):
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