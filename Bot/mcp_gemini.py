# import os
# import asyncio
# import json
# from google import genai
# from google.genai import types
from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from dotenv import load_dotenv
#
# load_dotenv()
#
# # Initialize Gemini client
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# messages = []
# # Configure your local MCP server (seat map server instead of flight search)
server_params = StdioServerParameters(
    command="python",   # or the CLI you use to run your MCP server
    args=["mcp_server.py", "--connection_type", "stdio"],
    env={},  # Add env vars if your MCP server needs any
)
#
# async def chat_loop():
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             print("✅ Connected to MCP SeatMap server")
#
#             # Dynamically discover available tools
#             mcp_tools = await session.list_tools()
#             tools = [
#                 types.Tool(
#                     function_declarations=[{
#                         "name": tool.name,
#                         "description": tool.description,
#                         "parameters": {
#                             k: v
#                             for k, v in tool.inputSchema.items()
#                             if k not in ["additionalProperties", "$schema"]
#                         },
#                     }]
#                 )
#                 for tool in mcp_tools.tools
#             ]
#
#             while True:
#                 user_query = input("\n🎤 You: ")
#                 if user_query.lower() in ["exit", "quit"]:
#                     print("👋 Goodbye!")
#                     break
#
#                 messages.append(user_query)
#
#                 # Ask Gemini with tools
#                 response = client.aio.models.generate_content(
#                     model="gemini-2.5-flash",
#                     contents=user_query,
#                     config=types.GenerateContentConfig(
#                         temperature=0,
#                         tools=tools,
#                     ),
#                 )
#                 messages.append(response.candidates[0].content)
#                 print(f"\n\n\nRESPONSE=> {response.text}")
#
#                 # Step 2: Handle tool call
#                 final_answer = None
#                 # Check if Gemini suggested a tool call
#                 if response.candidates and response.candidates[0].content.parts:
#
#                     for part in response.candidates[0].content.parts:
#                         if part.function_call:
#                             fn = part.function_call
#                             print(f"🤖 Gemini suggests tool: {fn.name} with {fn.args}")
#
#                             # Run the tool
#                             result = await session.call_tool(
#                                 fn.name, arguments=dict(fn.args)
#                             )
#                             print("📊 Tool result:", result)
#                             print(result.model_dump())
#
#                             # Step 3: Feed tool result back to Gemini
#                             followup = client.models.generate_content(
#                                 model="gemini-2.5-pro-exp-03-25",
#                                 contents=[
#                                     {"role": "user", "parts": [user_query]},
#                                     {"role": "model", "parts": [part]},
#                                     {"role": "user", "parts": [f"Tool response: {json.dumps(result.model_dump())}"]},
#                                 ],
#                                 config=types.GenerateContentConfig(
#                                     temperature=0
#                                 ),
#                             )
#                             final_answer = followup.text
#                             break
#
#                         # Step 4: If no tool was used, just take Gemini's answer
#                     if not final_answer:
#                         final_answer = response.text
#
#                     print("🤖 Gemini:", final_answer)
#
# if __name__ == "__main__":
#     asyncio.run(chat_loop())

"""
MCP Client using Gemini
-----------------------
This module defines the MCP_ChatBot class, which connects to an MCP server and uses Google Gemini 2.0 Flash LLM for conversational AI with tool/function calling support.

Features:
- Loads environment variables for API keys and configuration.
- Connects to an MCP server using stdio and lists available tools.
- Uses Gemini 2.0 Flash for LLM-powered chat, with automatic function/tool calling via MCP tools.
- Supports an interactive chat loop for user queries.
- Handles tool invocation and integrates tool results into the conversation.

Classes:
    MCP_ChatBot: Main chatbot class for managing the Gemini LLM and MCP tool integration.

Usage:
    Run this script directly to start the chatbot and connect to the MCP server.
"""

from dotenv import load_dotenv
from contextlib import AsyncExitStack
from typing import List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig
import os
import asyncio
import nest_asyncio

nest_asyncio.apply()
_ = load_dotenv()


class MCP_ChatBot:
    def __init__(self):
        """
        Initialize the MCP_ChatBot instance.
        Sets up the Gemini client, model, tool configuration, and message history.
        """
        # Initialize session and client objects
        self.session: ClientSession = None
        self.exit_stack = AsyncExitStack()
        self.llm = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = "gemini-2.5-flash"
        self.available_tools: List[dict] = []
        self.content_config: GenerateContentConfig
        self.messages: List[types.Content] = []

    async def process_query(self, query: str):
        """
        Process a user query by sending it to the Gemini LLM and handling tool calls.
        Appends the user message to the conversation, sends it to Gemini, and prints the response.
        If Gemini triggers a tool call, the MCP session is used to execute the tool and the result is integrated into the conversation.

        Args:
            query (str): The user's input query.
        """
        # format user message as structured Content object for Gemini
        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)]  # converts text query to Gemini format
        )
        self.messages.append(message)

        # Send request to the model with MCP function declarations
        response = await self.llm.aio.models.generate_content(
            model=self.model,
            contents=self.messages,
            config=genai.types.GenerateContentConfig(
                temperature=0,
                tools=[self.session],  # uses the session, will automatically call the tool
                # Uncomment if you **don't** want the sdk to automatically call the tool
                # automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                #     disable=True
                # ),
            ),
        )
        self.messages.append(response.candidates[0].content)
        print(response.text)

    async def chat_loop(self):
        """
        Run an interactive chat loop for user input.
        Continuously prompts the user for queries, processes them, and prints responses until 'quit' is entered.
        """
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                await self.process_query(query)
                print("\n")

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server_and_run(self):
        """
        Connect to the MCP server, initialize the session, list available tools, and start the chat loop.
        Sets up the stdio connection to the MCP server and prepares the chatbot for interaction.
        """
        # Create server parameters for stdio connection
        # server_params = StdioServerParameters(
        #     command="uv",  # Executable
        #     args=["run", "research_server.py"],  # Optional command line arguments
        #     env=None,  # Optional environment variables
        # )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session

                # Initialize the connection between client and server
                await session.initialize()

                # List available tools
                response = await session.list_tools()

                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools], "\n")
                print(response.tools)
                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
