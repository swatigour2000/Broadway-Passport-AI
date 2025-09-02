import os
import sys
from pathlib import Path

# Set the project root directory path
PROJECT_ROOT = Path(__file__).parent.parent  # Adjust based on your structure
sys.path.append(str(PROJECT_ROOT))

import json
from Bot.mcp_client import MCPAgent
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Union
import time
import uuid
import platform
import hashlib
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime, timedelta
from fastapi import Request
import aiofiles

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add this to your imports
import websockets

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("app.log"),  # Log to a file
    ]
)
# Add parent directory to system path for imports

app = FastAPI(root_path="/agent-api",  # Keep the agent-api prefix for all routes
              #    docs_url="/agent-api/docs",   # Set the docs URL to /agent-api/docs
              redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can specify domains as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


# Initialize agent at startup
@app.on_event("startup")
async def startup_event():
    global agent
    agent = MCPAgent()
    await agent.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    global agent
    if agent:
        await agent.close()


# API_TOKEN = "doehfvr345niniouh676nkbibib54"
#
# def authenticate(api_token: str = Depends(lambda: API_TOKEN)):
#     if api_token != API_TOKEN:
#         raise HTTPException(status_code=401, detail="Unauthorized: Invalid API token")

# Add this class for WebSocket connections management
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()
class StreamOptions(BaseModel):
    # Example fields (adjust based on your API requirements)
    chunk_size: Optional[int] = 4096  # Just an example, set to whatever is relevant
    timeout: Optional[int] = 30

class FunctionCall(BaseModel):
    type: str  # The type of the function, e.g., "function"
    function: Dict[str, str]

class Message(BaseModel):
    role: str  # Role of the message sender (e.g., "user", "assistant", "system")
    content: str  # Content of the message
    tool_calls: Optional[list] = None

class CompletionRequest(BaseModel):
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = 100  # Default value
    messages: List[Message]
    model: str
    n: Optional[int] = 1
    parallel_tool_calls: Optional[bool] = True
    presence_penalty: Optional[float] = 0
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    service_tier: Optional[str] = "on_demand"
    stop: Optional[str] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = 1  # Default value
    tool_choice: Optional[Union[str, FunctionCall, None]] = None
    tools: Optional[List[FunctionCall]] = None
    top_logprobs: Optional[int] = None  # Integer between 0 and 20 or None
    user: Optional[str] = None
    top_p: Optional[float] = 1.0  # Default value

class Choice(BaseModel):
    index: int  # Index of the choice
    message: Optional[Message] = None  # The message object containing the response content
    delta: Optional[Union[Message, Dict, None]] = None
    logprobs: Optional[Dict] = None  # Logprobs (optional)
    finish_reason: Optional[str] = None  # Reason for completion termination (e.g., "stop")

class Usage(BaseModel):
    prompt_tokens: int  # The number of tokens used for the prompt
    completion_tokens: int  # The number of tokens used for the completion
    total_tokens: int  # The total number of tokens used
    prompt_token_details: Optional[Union[Dict, None]]
    completion_token_details: Optional[Union[Dict, None]]

class CompletionResponse(BaseModel):
    id: str
    choices: List[Choice]
    object: str
    created: int
    model: str
    service_tier: Optional[str] = None  # The service tier used for processing the request, optional
    system_fingerprint: str  # A fingerprint representing the backend configuration
    object: str  # Always "chat.completion" in this case
    usage: Usage

class ChunkResponse(BaseModel):
    id: str  # Unique identifier for the chat completion (same for all chunks)
    choices: List[Choice]  # List of chat completion choices (could be empty in the last chunk)
    created: int  # Unix timestamp (in seconds) of when the chunk was created (same for all chunks)
    model: str  # The model used to generate the completion
    service_tier: Optional[str] = None  # The service tier used for processing the request, optional
    system_fingerprint: str  # A fingerprint representing the backend configuration
    object: str = "chat.completion.chunk"  # Always "chat.completion.chunk"
    usage: Optional[Usage] = None

class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class ModelListResponse(BaseModel):
    object: str
    data: List[Model]


def generate_unique_id():
    return str(uuid.uuid4()).replace("-", "")  # Remove dashes for a compact alphanumeric ID

def generate_system_fingerprint():
    # Collect system information
    system_info = f"{platform.system()}_{platform.node()}_{platform.processor()}_{platform.machine()}_{os.environ.get('HOSTNAME', '')}"

    # Generate a hash of the system info to create a consistent fingerprint
    fingerprint = hashlib.sha256(system_info.encode('utf-8')).hexdigest()

    return fingerprint


async def async_write_log(log_entry: dict):
    """Asynchronously write log entry with file locking"""
    log_file = f"{LOG_DIR}/api_requests.log"
    try:
        # Ensure timestamp exists
        if "timestamp" not in log_entry:
            log_entry["timestamp"] = datetime.now().isoformat()

        async with aiofiles.open(log_file, mode='a') as f:
            await f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logging.error(f"Async log write failed: {str(e)}")
        # Fallback to synchronous write
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as sync_e:
            logging.error(f"Sync log write also failed: {str(sync_e)}")
@app.get("/")
async def home():
    return "Welcome to Broadway Passport-Bot Home..."


@app.get("/v2/models")
async def get_models():
    """List all available models."""
    return ModelListResponse(
        object="list",
        data=[
            Model(
                id="BroadwayPass-Ai:Latest",
                object="model",
                created=1689523547,
                owned_by="Broadway Passport Team"
            ),
            # Model(
            #     id="SocialNest-Ai:Latest",
            #     object="model",
            #     created=1689523547,
            #     owned_by="Nest AI Team"
            # ),
            # Model(
            #     id="PremiumNest-Ai:Latest",
            #     object="model",
            #     created=1689523547,
            #     owned_by="Nest AI Team"
            # )
        ]
    )

@app.get("/v2/models/{model_id}", response_model=Model)
def retrieve_model(model_id: str):
    """Retrieve details of a specific model."""
    if model_id == "CPT-Ai:Latest":
        Model(
            id="BroadwayPass-Ai:Latest",
            object="model",
            created=1689523547,
            owned_by="Broadway Passport Team"
        )

    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_id}' not found. Available model: 'GeoNest-Ai:Latest'"
    )








@app.post("/v2/chat/completions", response_model=CompletionResponse)
async def generate_completion(request: CompletionRequest):
    try:
        # Extract request data
        messages = request.messages
        model = request.model
        max_tokens = request.max_tokens
        temperature = request.temperature
        top_p = request.top_p
        stream = request.stream
        print(messages)

        # Process input and generate context
        if not messages:
            raise HTTPException(status_code=400, detail="Messages are required.")

        if model not in ["BroadwayPass-Ai:Latest"]:
            raise HTTPException(status_code=400, detail="Invalid model selection. Please choose a valid model.")


        messages = [message.dict() for message in messages]

        if stream:
            async def generate():
                async for chunk in agent.stream_process(messages):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
        else:

            print(f"**** {stream}")

            response_text = await agent.process(messages)
            usage = response_text[1]
            print("#####", response_text)

            # prompt_token = count_prompt_tokens(messages)
            # Create a response based on the expected structure
            response = CompletionResponse(
                id=generate_unique_id(),
                object="chat.completion",
                created=int(time.time()),
                model=model,
                choices=response_text[0],
                usage=Usage(
                    prompt_tokens=usage["input_tokens"],  # Example: count prompt tokens
                    completion_tokens=usage["output_tokens"],  # Example: count completion tokens
                    total_tokens=usage["total_tokens"],
                    prompt_token_details=usage["input_token_details"],
                    completion_token_details=usage["output_token_details"]
                ),
                system_fingerprint=generate_system_fingerprint()  # Can be dynamically generated if needed
            )
            return response

    except Exception as e:
        logging.error(f"Error in generate_completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Add this endpoint for the log viewer page
@app.get("/v2/logs/view", response_class=HTMLResponse)
async def get_log_viewer():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time BroadwayPassport Bot API Logs</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f5f5f5; 
            }
            #logs { 
                background: white; 
                border: 1px solid #ddd; 
                border-radius: 4px; 
                padding: 20px; 
                height: 80vh; 
                overflow-y: auto; 
            }
            .log-entry { 
                margin-bottom: 15px; 
                padding: 15px; 
                border-left: 4px solid #4285f4; 
                background: #f8f9fa; 
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .log-entry.error { border-left-color: #ea4335; }
            .log-entry.warning { border-left-color: #fbbc05; }
            .log-entry.success { border-left-color: #34a853; }
            .log-header { 
                display: flex; 
                justify-content: space-between;
                margin-bottom: 8px;
            }
            .log-title {
                font-weight: bold;
                font-size: 1.1em;
            }
            .timestamp { 
                color: #666; 
                font-size: 0.8em; 
            }
            .method { 
                font-weight: bold; 
                color: #4285f4; 
                margin-right: 8px;
            }
            .path { 
                font-weight: bold;
                margin-right: 8px;
            }
            .ip { 
                color: #666;
                font-size: 0.9em;
            }
            .status-badge {
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: bold;
                color: white;
            }
            .status-2xx { background-color: #34a853; }
            .status-4xx { background-color: #fbbc05; }
            .status-5xx { background-color: #ea4335; }
            .duration {
                color: #555;
                font-size: 0.9em;
            }
            .error-message { 
                margin-top: 8px;
                padding: 8px;
                background-color: #fee;
                border-left: 3px solid #ea4335;
                border-radius: 3px;
            }
            details {
                margin-top: 10px;
            }
            summary {
                cursor: pointer;
                font-weight: bold;
                padding: 5px;
                background: #eee;
                border-radius: 3px;
            }
            .section {
                margin-top: 10px;
            }
            .section-title {
                font-weight: bold;
                margin-bottom: 5px;
            }
            pre {
                background: #f8f8f8;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
                margin: 5px 0;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }
            .json-key { color: #d14; }
            .json-string { color: #690; }
            .json-number { color: #08c; }
            .json-boolean { color: #08c; }
            .json-null { color: #08c; }
        </style>
    </head>
    <body>
        <h1>Real-time Nest AI API Logs</h1>
        <div id="logs"></div>

        <script>
            // Syntax highlighting for JSON
            function syntaxHighlight(json) {
                if (typeof json != 'string') {
                    json = JSON.stringify(json, null, 2);
                }
                json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                return json.replace(
                    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
                    function (match) {
                        let cls = 'json-number';
                        if (/^"/.test(match)) {
                            if (/:$/.test(match)) {
                                cls = 'json-key';
                            } else {
                                cls = 'json-string';
                            }
                        } else if (/true|false/.test(match)) {
                            cls = 'json-boolean';
                        } else if (/null/.test(match)) {
                            cls = 'json-null';
                        }
                        return '<span class="' + cls + '">' + match + '</span>';
                    }
                );
            }

            const logsDiv = document.getElementById('logs');
            const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
            const ws = new WebSocket(`${wsProtocol}${window.location.host}/agent-api/v2/logs/ws`);

            ws.onmessage = function(event) {
                try {
                    const log = JSON.parse(event.data);
                    const entry = document.createElement('div');

                    // Determine status class
                    const statusClass = 
                        log.status >= 500 ? 'error' : 
                        log.status >= 400 ? 'warning' : 'success';

                    // Determine status badge class
                    const statusBadgeClass = 
                        log.status >= 500 ? 'status-5xx' : 
                        log.status >= 400 ? 'status-4xx' : 'status-2xx';

                    entry.className = `log-entry ${statusClass}`;

                    // Format timestamp
                    const timestamp = new Date(log.timestamp);
                    const formattedTime = timestamp.toLocaleTimeString();
                    const formattedDate = timestamp.toLocaleDateString();

                    // Format request body if exists
                    let requestBody = '';
                    if (log.details.body) {
                        try {
                            const parsedBody = JSON.parse(log.details.body);
                            requestBody = syntaxHighlight(parsedBody);
                        } catch {
                            requestBody = log.details.body;
                        }
                    }

                    // Format response if exists
                    let responseBody = '';
                    if (log.response) {
                        try {
                            const parsedResponse = typeof log.response === 'string' ? 
                                JSON.parse(log.response) : log.response;
                            responseBody = syntaxHighlight(parsedResponse);
                        } catch {
                            responseBody = log.response;
                        }
                    }

                    entry.innerHTML = `
                        <div class="log-header">
                            <div>
                                <span class="method">${log.method}</span>
                                <span class="path">${log.path}</span>
                                <span class="ip">from ${log.client_ip}</span>
                            </div>
                            <div>
                                <span class="status-badge ${statusBadgeClass}">${log.status}</span>
                                <span class="duration">${log.duration_ms}ms</span>
                            </div>
                        </div>
                        <div class="timestamp">${formattedDate} ${formattedTime}</div>

                        ${log.error ? `
                        <div class="error-message">
                            <strong>Error:</strong> ${log.error}
                        </div>
                        ` : ''}

                        <details>
                            <summary>Request Details</summary>
                            <div class="section">
                                <div class="section-title">Headers</div>
                                <pre>${syntaxHighlight(log.details.headers)}</pre>
                            </div>

                            ${log.details.query_params && Object.keys(log.details.query_params).length > 0 ? `
                            <div class="section">
                                <div class="section-title">Query Parameters</div>
                                <pre>${syntaxHighlight(log.details.query_params)}</pre>
                            </div>
                            ` : ''}

                            ${log.details.path_params && Object.keys(log.details.path_params).length > 0 ? `
                            <div class="section">
                                <div class="section-title">Path Parameters</div>
                                <pre>${syntaxHighlight(log.details.path_params)}</pre>
                            </div>
                            ` : ''}

                            ${requestBody ? `
                            <div class="section">
                                <div class="section-title">Request Body</div>
                                <pre>${requestBody}</pre>
                            </div>
                            ` : ''}
                        </details>

                        ${responseBody ? `
                        <details>
                            <summary>Response</summary>
                            <pre>${responseBody}</pre>
                        </details>
                        ` : ''}
                    `;

                    logsDiv.appendChild(entry);
                    logsDiv.scrollTop = logsDiv.scrollHeight;
                } catch (e) {
                    console.error("Error processing log message:", e);
                }
            };

            ws.onclose = function() {
                const entry = document.createElement('div');
                entry.className = 'log-entry warning';
                entry.textContent = 'WebSocket connection closed. Refresh page to reconnect.';
                logsDiv.appendChild(entry);
            };
        </script>
    </body>
    </html>
    """

@app.websocket("/v2/logs/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = None
    error = None
    request_body = None
    response_body = None

    try:
        # Store the request body if it's a POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                request_body = await request.body()
                # Create a new request with the stored body
                request._body = request_body
            except Exception as e:
                logging.error(f"Error reading request body: {str(e)}")
                request_body = None

        # Process the request
        response = await call_next(request)

        # Handle streaming responses differently
        if isinstance(response, StreamingResponse):
            # For streaming responses, we'll create a wrapper to capture the content
            original_iterator = response.body_iterator

            async def logging_wrapper():
                chunks = []
                async for chunk in original_iterator:
                    chunks.append(chunk)
                    yield chunk

                # After streaming is complete, store the response
                nonlocal response_body
                response_body = b''.join(chunks).decode('utf-8')
                try:
                    response_body = json.loads(response_body)
                except:
                    pass

            response.body_iterator = logging_wrapper()
        else:
            # For non-streaming responses
            try:
                if hasattr(response, 'body'):
                    if isinstance(response.body, bytes):
                        response_body = response.body.decode('utf-8')
                    else:
                        response_body = str(response.body)
                else:
                    response_body = "[Response body not available]"

                # Try to parse JSON if it looks like JSON
                if response_body and isinstance(response_body, str) and response_body.startswith(('{', '[')):
                    try:
                        response_body = json.loads(response_body)
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logging.error(f"Error capturing response body: {str(e)}")
                response_body = f"[Error capturing response: {str(e)}]"

    except HTTPException as http_exc:
        error = str(http_exc.detail)
        response = JSONResponse(
            content={"detail": error},
            status_code=http_exc.status_code
        )
        response_body = {"detail": error}
    except Exception as e:
        error = str(e)
        response = JSONResponse(
            content={"detail": "Internal server error"},
            status_code=500
        )
        response_body = {"detail": "Internal server error"}
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        # Calculate duration
        duration_ms = round((time.time() - start_time) * 1000, 2)

        # Prepare log data
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "status": response.status_code if response else 500,
            "duration_ms": duration_ms,
            "error": error,
            "response": response_body,
            "details": {
                "headers": dict(request.headers),
                "query_params": dict(request.query_params),
                "path_params": request.path_params,
                "body": request_body.decode('utf-8') if request_body else None,
            }
        }

        # Broadcast to WebSocket clients
        try:
            await manager.broadcast(json.dumps(log_data, default=str))
        except Exception as e:
            logging.error(f"WebSocket broadcast failed: {str(e)}")

        # Also log to file
        try:
            await async_write_log(log_data)
        except Exception as e:
            logging.error(f"Async log write failed: {str(e)}")

    return response


# Helper class for streaming responses
class AsyncIteratorWrapper:
    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


@app.get("/v2/logs/last24hours")
async def get_last_24h_logs():
    """
    Retrieve logs from the last 24 hours with improved error handling
    """
    log_file = Path(f"{LOG_DIR}/api_requests.log")
    current_time = datetime.now()
    twenty_four_hours_ago = current_time - timedelta(hours=24)
    logs = []

    try:
        # Debug: Check if log directory exists
        if not Path(LOG_DIR).exists():
            raise HTTPException(
                status_code=500,
                detail=f"Log directory {LOG_DIR} does not exist"
            )

        # Debug: Check if log file exists
        if not log_file.exists():
            return {
                "logs": [],
                "count": 0,
                "debug": {
                    "log_file": str(log_file),
                    "status": "File does not exist"
                }
            }

        # Read file synchronously for better error handling
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read log file: {str(e)}"
            )

        # Process logs in reverse order
        for line in reversed(lines):
            if not line.strip():
                continue

            try:
                log_entry = json.loads(line)

                # Debug: Check if timestamp exists
                timestamp = log_entry.get("timestamp")
                if not timestamp:
                    continue

                # Parse timestamp with multiple format support
                try:
                    if isinstance(timestamp, (int, float)):
                        # Handle Unix timestamp
                        log_time = datetime.fromtimestamp(timestamp)
                    elif isinstance(timestamp, str):
                        # Try ISO format first
                        try:
                            log_time = datetime.fromisoformat(timestamp)
                        except ValueError:
                            # Fallback to other common formats
                            for fmt in [
                                "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO with timezone
                                "%Y-%m-%dT%H:%M:%S.%f",  # ISO without timezone
                                "%Y-%m-%d %H:%M:%S.%f",  # Space separator
                                "%Y-%m-%dT%H:%M:%S",  # ISO without microseconds
                                "%Y-%m-%d %H:%M:%S"  # Space without microseconds
                            ]:
                                try:
                                    log_time = datetime.strptime(timestamp, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                raise ValueError("No valid format found")
                    else:
                        continue

                    # Only include logs from last 24 hours
                    if log_time >= twenty_four_hours_ago:
                        logs.append(log_entry)
                    else:
                        break

                except Exception as e:
                    print(f"Skipping entry due to timestamp parsing error: {e}")
                    continue

            except json.JSONDecodeError as e:
                print(f"Skipping malformed log entry: {line}. Error: {e}")
                continue

        return {
            "logs": logs,
            "count": len(logs),
            "debug": {
                "log_file": str(log_file),
                "earliest_timestamp": twenty_four_hours_ago.isoformat(),
                "current_time": current_time.isoformat(),
                "entries_processed": len(lines),
                "valid_entries": len(logs)
            }
        }

    except Exception as e:
        print(f"Error in get_last_24h_logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving logs: {str(e)}"
        )

