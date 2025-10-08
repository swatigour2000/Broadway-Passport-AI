import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessageChunk
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import openai
import httpx
from contextlib import asynccontextmanager
from langchain_core.tools import StructuredTool
import time
from typing import AsyncGenerator, List, Dict, Any

# Load environment variables
load_dotenv()


class MCPAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("GPT_API_KEY"))
        self.server_params = StdioServerParameters(
            command="python",
            args=[r"mcp_server.py"],
        )
        self.server_url = "http://localhost:8666/sse"
        # self.model.bind_tools(tools=[{"type": "web_search"}])
        self.session = None
        self.tools = None
        self.agent = None
        self.read = None
        self.write = None
        self.client = None
        self.system_prompt = None
        self.analyze_screenshot_system_prompt = os.getenv("ANALYZE_SCREENSHOT_SYSTEM_PROMPT")
        self._openai_client = openai.AsyncOpenAI(api_key=os.getenv("GPT_API_KEY"))

    def _prepare_messages(self, messages: List[Dict], use_vision_prompt: bool = False) -> List[Dict]:
        """Prepare messages with appropriate system prompt"""
        system_prompt = self.analyze_screenshot_system_prompt if use_vision_prompt else self.system_prompt

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_prompt
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return messages

    def _contains_images(self, messages: List[Dict]) -> bool:
        """Check if messages contain any images"""
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            return True
                elif isinstance(content, str) and "data:image" in content:
                    return True
        return False
    async def web_search_tool(self, query: str):
        """Do the real time web search for all type of queries, tax rates, safety concerns, etc"""
        if not hasattr(self, '_openai_client'):
            self._openai_client = openai.AsyncOpenAI(
                api_key=os.getenv("GPT_API_KEY"),
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_keepalive_connections=10),
                    timeout=20
                )
            )
        try:
            response = await self._openai_client.responses.create(
                model="gpt-4o-mini",
                tools=[{
                    "type": "web_search"
                }],
                input=query
            )
            raw_data = response.model_dump()
            output = raw_data.get("output", [{}])
            main_content = output[-1].get("content", [{}])[0].get("text", "") if len(output) > 0 else ""
            citations = (
                f"[{i + 1}] {ann['url']}" for i, ann in
            enumerate(output[-1].get("content", [{}])[0].get("annotations", [])) if ann.get('type') == "url_citation")

            content = "\n".join([main_content, *(["\n\nSources:"] + list(citations))]) if citations else main_content

            return (content, {
                "raw": raw_data,
                "metadata": {
                    "model": raw_data.get("model"),
                    "tokens": raw_data.get("usage", {})
                }
            })

        except Exception as e:
            return (f"Search error: {str(e)}", {"error": str(e)})

    async def initialize(self):
        """Initialize the MCP connection and create the agent"""
        try:
            # self.client = stdio_client(self.server_params)
            self.client = MultiServerMCPClient(
                {
                    "masterapi_db": {
                        "url": self.server_url,
                        "transport": "sse",
                    }
                }
            )

            self.tools = await self.client.get_tools()
            print("Tools loaded:", self.tools)

            # self.read, self.write = await self.client.__aenter__()
            # self.session = ClientSession(self.read, self.write)
            # await self.session.__aenter__()
            # await self.session.initialize()
            print("Session initialized.")

            # self.tools = await load_mcp_tools(self.session)
            # self.tools = await self.client.get_tools()
            print("Tools loaded:", self.tools)
            web_search_tool = StructuredTool(
                name='web_search_preview',
                description='Perform web searches for current information',
                args_schema={
                    'properties': {'query': {'title': 'Query', 'type': 'string'}},
                    'required': ['query'],
                    'title': 'WebSearchArguments',
                    'type': 'object'
                },
                coroutine=self.web_search_tool,
                response_format='content_and_artifact'
            )
            self.tools.append(web_search_tool)
            self.system_prompt = os.getenv("CPT_BOT_SYSTEM_PROMPT")

            self.agent = create_react_agent(self.model, tools=self.tools)
            print("Agent created.")

        except Exception as e:
            print(f"Initialization error: {e}")
            await self.close()
            raise

    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self.client:
            await self.client.__aexit__(None, None, None)
        self.session = None
        self.client = None
        self.read = None
        self.write = None

    async def stream_process(self, messages, model="GeoNest-Ai:Latest", use_vision_prompt: bool = False) -> AsyncGenerator[dict, None]:
        """Stream messages with tool calls and agent responses in OpenAI format"""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        try:
            prepared_messages = self._prepare_messages(messages, use_vision_prompt)

            response = self.agent.astream({"messages": prepared_messages}, stream_mode="messages")
            full_content = ""
            current_tool_call = None
            # tool_call_id = f"call_{int(time.time())}"

            async for chunk in response:
                print("---------", chunk)
                if not isinstance(chunk[0], AIMessageChunk):
                    continue

                message = chunk[0]
                metadata = chunk[1] if len(chunk) > 1 else {}

                # Handle tool call streaming
                if hasattr(message, 'tool_call_chunks') and message.tool_call_chunks:
                    for tool_chunk in message.tool_call_chunks:
                        if current_tool_call is None:
                            current_tool_call = {
                                # "id": tool_call_id,
                                "name": tool_chunk.get('name', 'unknown_tool'),
                                "args": tool_chunk.get('args', ''),
                                "index": tool_chunk.get('index', 0)
                            }
                        else:
                            current_tool_call["args"] += tool_chunk.get('args', '')

                        # Yield tool call chunk in OpenAI format
                        yield {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "system_fingerprint": metadata.get('system_fingerprint', 'fp_unknown'),
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [{
                                        "index": current_tool_call["index"],
                                        # "id": current_tool_call["id"],
                                        "function": {
                                            "name": current_tool_call["name"],
                                            "arguments": current_tool_call["args"]
                                        },
                                        "type": "function"
                                    }]
                                },
                                "logprobs": None,
                                "finish_reason": None
                            }]
                        }

                # Handle content streaming
                elif message.content:
                    full_content += message.content
                    current_tool_call = None  # Reset tool call tracking

                    # Yield content chunk in OpenAI format
                    yield {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "system_fingerprint": metadata.get('system_fingerprint', 'fp_unknown'),
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": message.content
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }

                # Handle final message with usage data
                if getattr(message, 'response_metadata', {}).get('finish_reason'):
                    yield {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "system_fingerprint": metadata.get('system_fingerprint', 'fp_unknown'),
                        # "usage": {
                        #     "prompt_tokens": metadata.get('prompt_tokens', 0),
                        #     "completion_tokens": metadata.get('completion_tokens', 0),
                        #     "total_tokens": metadata.get('total_tokens', 0)
                        # },
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": message.response_metadata['finish_reason']
                        }]
                    }

        except Exception as agent_error:
            print(f"Error during agent invocation: {agent_error}")
            raise agent_error



    async def process(self, messages: List[Dict], use_vision_prompt: bool = False) -> tuple:
        """Process messages using the pre-initialized agent"""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        try:
            prepared_messages = self._prepare_messages(messages, use_vision_prompt)

            response = self.agent.astream({"messages": prepared_messages}, stream_mode="updates")
            # messages = response["messages"]
            print(response)
            res = ''
            print("\n\n+++++++++++++++++++++++++++")
            print("Assistant: ", end="", flush=True)
            choices = []
            final_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_token_details": {"audio": 0, "cache_read": 0},
                "output_token_details": {"audio": 0, "reasoning": 0}
            }

            def update_usage(usage_data):
                """Helper function to update token usage"""
                final_usage["input_tokens"] += usage_data.get("prompt_tokens", 0)
                final_usage["output_tokens"] += usage_data.get("completion_tokens", 0)
                final_usage["total_tokens"] += usage_data.get("total_tokens", 0)

                input_details = usage_data.get("prompt_tokens_details", {})
                final_usage["input_token_details"]["audio"] += input_details.get("audio_tokens", 0)
                final_usage["input_token_details"]["cache_read"] += input_details.get("cached_tokens", 0)

                output_details = usage_data.get("completion_tokens_details", {})
                final_usage["output_token_details"]["audio"] += output_details.get("audio_tokens", 0)
                final_usage["output_token_details"]["reasoning"] += output_details.get("reasoning_tokens", 0)

            async for chunk in response:

                print("**", chunk)
                if "agent" not in chunk or not chunk["agent"].get("messages"):
                    continue  # Skip invalid chunks
                message = chunk["agent"]["messages"][0]  # First message from agent
                usage_data = message.response_metadata.get("token_usage", {})

                if message.content == "" and "tool_calls" in message.additional_kwargs:
                    tool_call = message.additional_kwargs["tool_calls"]
                    # tool_name = tool_call["function"]["name"]
                    # tool_args = tool_call["function"]["arguments"]

                    choices.append({
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",  # Assistant doesn't return content for tool calls
                            "tool_calls": tool_call
                        },
                        "logprobs": None,
                        "finish_reason": "tool_calls"
                    })
                    update_usage(usage_data)

                elif message.content:
                    print(message.content, end="", flush=True)
                    choices.append({
                        "index": 0,  # Typically, only one response per request
                        "message": {
                            "role": "assistant",
                            "content": message.content
                        },
                        "logprobs": None,
                        "finish_reason": "stop"
                    })
                    res += message.content
                    update_usage(usage_data)
            print(choices, final_usage)
            return (choices, final_usage)
            print("\n\n+++++++++++++++++++")
            print("Agent Response:", choices, final_usage)
            return res

        except Exception as agent_error:
            print(f"Error during agent invocation: {agent_error}")
            raise agent_error


async def main():
    agent = MCPAgent()
    await agent.initialize()
    messages = []

    try:
        while True:
            try:
                # User input
                query = input("\nYou: ").strip()
                if query.lower() == "exit":
                    print("\nGoodbye!")
                    break

                # Add the user message to history
                user_message = {"role": "user", "content": query}
                messages.append(user_message)

                # Non streaming
                response, usage = await agent.process(messages)
                print("\nAssistant:", response, usage)

                # #streaming
                # response = ''
                # async for chunk in agent.stream_process(messages):
                #     print(chunk)
                #     # Process the OpenAI-format chunks
                #     if chunk['choices'][0]['delta'].get('content'):
                #         print(chunk['choices'][0]['delta']['content'], end="", flush=True)
                #         response += chunk['choices'][0]['delta']['content']
                #     if chunk['choices'][0]['delta'].get('tool_calls'):
                #         for tool_call in chunk['choices'][0]['delta']['tool_calls']:
                #             print(f"\n[Tool Call: {tool_call['function']['name']}]")
                #             print(f"[Arguments: {tool_call['function']['arguments']}]")
                #
                #     if chunk['choices'][0].get('finish_reason'):
                #         print(f"\n[Finished: {chunk['choices'][0]['finish_reason']}]")
                #         # print(f"[Token usage: {chunk.get('usage', {})}]")
                # print("\nAssistant:", response)
                messages.append({"role": "assistant", "content": response[-1]["message"]["content"]})

            except Exception as e:
                print("Error:", e)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())