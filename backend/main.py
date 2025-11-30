"""
Python Backend for AI Computer Use with Kernel SDK and NVIDIA Llama 4
Provides WebSocket-based real-time streaming for AI browser control
"""

import asyncio
import json
import base64
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Hardcoded API keys
NVIDIA_API_KEY = "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"
KERNEL_API_KEY = "sk_0fc1b266-d4ea-40b1-9428-c5e91fe33715.zmMuKnvrhYkChScdXVBk7Z98xHh2p/1iSt6OrCwgL/Y"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kernel SDK client
class KernelBrowserClient:
    """Client for Kernel SDK browser control"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.onkernel.com"
        self.session_id: Optional[str] = None
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0
        )
    
    async def create_browser(self) -> Dict[str, Any]:
        """Create a new browser session"""
        response = await self.client.post(f"{self.base_url}/v1/browsers")
        response.raise_for_status()
        data = response.json()
        self.session_id = data.get("session_id")
        return data
    
    async def get_stream_url(self) -> Optional[str]:
        """Get the stream URL for the browser session"""
        if not self.session_id:
            return None
        response = await self.client.get(f"{self.base_url}/v1/browsers/{self.session_id}/stream")
        response.raise_for_status()
        data = response.json()
        return data.get("url")
    
    async def click_mouse(self, x: int, y: int, button: str = "left", click_type: str = "click", num_clicks: int = 1, hold_keys: List[str] = None) -> Dict[str, Any]:
        """Click mouse at coordinates"""
        payload = {"x": x, "y": y, "button": button, "click_type": click_type, "num_clicks": num_clicks}
        if hold_keys:
            payload["hold_keys"] = hold_keys
        response = await self.client.post(
            f"{self.base_url}/v1/browsers/{self.session_id}/computer/click-mouse",
            json=payload
        )
        response.raise_for_status()
        return {"success": True, "action": f"Clicked {button} at ({x}, {y})"}
    
    async def move_mouse(self, x: int, y: int, hold_keys: List[str] = None) -> Dict[str, Any]:
        """Move mouse to coordinates"""
        payload = {"x": x, "y": y}
        if hold_keys:
            payload["hold_keys"] = hold_keys
        response = await self.client.post(
            f"{self.base_url}/v1/browsers/{self.session_id}/computer/move-mouse",
            json=payload
        )
        response.raise_for_status()
        return {"success": True, "action": f"Moved mouse to ({x}, {y})"}
    
    async def capture_screenshot(self, region: Dict[str, int] = None) -> bytes:
        """Capture screenshot of browser"""
        params = {}
        if region:
            params = region
        response = await self.client.get(
            f"{self.base_url}/v1/browsers/{self.session_id}/computer/screenshot",
            params=params
        )
        response.raise_for_status()
        return response.content
    
    async def type_text(self, text: str, delay: int = None) -> Dict[str, Any]:
        """Type text in browser"""
        payload = {"text": text}
        if delay:
            payload["delay"] = delay
        response = await self.client.post(
            f"{self.base_url}/v1/browsers/{self.session_id}/computer/type",
            json=payload
        )
        response.raise_for_status()
        return {"success": True, "action": f"Typed: {text}"}
    
    async def press_key(self, keys: List[str], duration: int = None, hold_keys: List[str] = None) -> Dict[str, Any]:
        """Press keys in browser"""
        payload = {"keys": keys}
        if duration:
            payload["duration"] = duration
        if hold_keys:
            payload["hold_keys"] = hold_keys
        response = await self.client.post(
            f"{self.base_url}/v1/browsers/{self.session_id}/computer/press-key",
            json=payload
        )
        response.raise_for_status()
        return {"success": True, "action": f"Pressed keys: {keys}"}
    
    async def scroll(self, x: int, y: int, delta_x: int = 0, delta_y: int = 0) -> Dict[str, Any]:
        """Scroll in browser"""
        payload = {"x": x, "y": y, "delta_x": delta_x, "delta_y": delta_y}
        response = await self.client.post(
            f"{self.base_url}/v1/browsers/{self.session_id}/computer/scroll",
            json=payload
        )
        response.raise_for_status()
        return {"success": True, "action": f"Scrolled at ({x}, {y}) by ({delta_x}, {delta_y})"}
    
    async def drag_mouse(self, path: List[List[int]], button: str = "left", delay: int = 0, steps_per_segment: int = 10, step_delay_ms: int = 50, hold_keys: List[str] = None) -> Dict[str, Any]:
        """Drag mouse along path"""
        payload = {
            "path": path,
            "button": button,
            "delay": delay,
            "steps_per_segment": steps_per_segment,
            "step_delay_ms": step_delay_ms
        }
        if hold_keys:
            payload["hold_keys"] = hold_keys
        response = await self.client.post(
            f"{self.base_url}/v1/browsers/{self.session_id}/computer/drag-mouse",
            json=payload
        )
        response.raise_for_status()
        return {"success": True, "action": f"Dragged along path with {len(path)} points"}
    
    async def close(self):
        """Close the browser session"""
        if self.session_id:
            try:
                await self.client.delete(f"{self.base_url}/v1/browsers/{self.session_id}")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
        await self.client.aclose()


# Computer tools definition for function calling
COMPUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click_mouse",
            "description": "Click the mouse at specific coordinates on the screen",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate to click"},
                    "y": {"type": "integer", "description": "Y coordinate to click"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Mouse button to click"},
                    "num_clicks": {"type": "integer", "description": "Number of clicks (1 for single, 2 for double)"}
                },
                "required": ["x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_mouse",
            "description": "Move the mouse cursor to specific coordinates",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate to move to"},
                    "y": {"type": "integer", "description": "Y coordinate to move to"}
                },
                "required": ["x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "capture_screenshot",
            "description": "Take a screenshot of the current browser screen to see what's displayed",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "object",
                        "description": "Optional region to capture",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "width": {"type": "integer"},
                            "height": {"type": "integer"}
                        }
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into the currently focused element",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                    "delay": {"type": "integer", "description": "Delay between keystrokes in milliseconds"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press keyboard keys (e.g., Enter, Tab, Ctrl+C)",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keys to press (e.g., ['Enter'], ['Ctrl+t'])"
                    }
                },
                "required": ["keys"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the page at a specific position",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate for scroll position"},
                    "y": {"type": "integer", "description": "Y coordinate for scroll position"},
                    "delta_x": {"type": "integer", "description": "Horizontal scroll amount"},
                    "delta_y": {"type": "integer", "description": "Vertical scroll amount (positive=down, negative=up)"}
                },
                "required": ["x", "y", "delta_y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drag_mouse",
            "description": "Drag the mouse along a path of points",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "description": "Array of [x, y] coordinate pairs defining the drag path"
                    },
                    "button": {"type": "string", "enum": ["left", "right"], "description": "Mouse button to hold during drag"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a specified duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number", "description": "Number of seconds to wait"}
                },
                "required": ["seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Call this when the task has been completed successfully",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of what was accomplished"}
                },
                "required": ["summary"]
            }
        }
    }
]


class NvidiaLlamaClient:
    """Client for NVIDIA Llama 4 API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=120.0
        )
        self.model = "meta/llama-4-maverick-17b-128e-instruct"
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        stream: bool = True
    ):
        """Create a chat completion with optional function calling"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": stream
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        if stream:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue
        else:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            yield response.json()
    
    async def close(self):
        await self.client.aclose()


# Session manager
class SessionManager:
    """Manages browser sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, KernelBrowserClient] = {}
    
    async def create_session(self) -> tuple[str, KernelBrowserClient]:
        """Create a new browser session"""
        client = KernelBrowserClient(KERNEL_API_KEY)
        browser_data = await client.create_browser()
        session_id = browser_data["session_id"]
        self.sessions[session_id] = client
        return session_id, client
    
    async def get_session(self, session_id: str) -> Optional[KernelBrowserClient]:
        """Get an existing session"""
        return self.sessions.get(session_id)
    
    async def close_session(self, session_id: str):
        """Close a session"""
        if session_id in self.sessions:
            await self.sessions[session_id].close()
            del self.sessions[session_id]
    
    async def close_all(self):
        """Close all sessions"""
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)


# Global session manager
session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting AI Computer Use Backend")
    yield
    logger.info("Shutting down - closing all sessions")
    await session_manager.close_all()


# Create FastAPI app
app = FastAPI(
    title="AI Computer Use Backend",
    description="Python backend for AI browser control with Kernel SDK and NVIDIA Llama 4",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionResponse(BaseModel):
    session_id: str
    stream_url: Optional[str]


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/session/create", response_model=CreateSessionResponse)
async def create_session():
    """Create a new browser session"""
    try:
        session_id, client = await session_manager.create_session()
        stream_url = await client.get_stream_url()
        return CreateSessionResponse(session_id=session_id, stream_url=stream_url)
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a browser session"""
    try:
        await session_manager.close_session(session_id)
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def execute_tool(client: KernelBrowserClient, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a computer tool and return the result"""
    try:
        if tool_name == "click_mouse":
            return await client.click_mouse(
                x=arguments["x"],
                y=arguments["y"],
                button=arguments.get("button", "left"),
                num_clicks=arguments.get("num_clicks", 1)
            )
        
        elif tool_name == "move_mouse":
            return await client.move_mouse(
                x=arguments["x"],
                y=arguments["y"]
            )
        
        elif tool_name == "capture_screenshot":
            image_data = await client.capture_screenshot(
                region=arguments.get("region")
            )
            base64_image = base64.b64encode(image_data).decode("utf-8")
            return {
                "success": True,
                "type": "image",
                "data": base64_image,
                "action": "Screenshot captured"
            }
        
        elif tool_name == "type_text":
            return await client.type_text(
                text=arguments["text"],
                delay=arguments.get("delay")
            )
        
        elif tool_name == "press_key":
            return await client.press_key(
                keys=arguments["keys"]
            )
        
        elif tool_name == "scroll":
            return await client.scroll(
                x=arguments["x"],
                y=arguments["y"],
                delta_x=arguments.get("delta_x", 0),
                delta_y=arguments["delta_y"]
            )
        
        elif tool_name == "drag_mouse":
            return await client.drag_mouse(
                path=arguments["path"],
                button=arguments.get("button", "left")
            )
        
        elif tool_name == "wait":
            await asyncio.sleep(min(arguments["seconds"], 5))
            return {"success": True, "action": f"Waited {arguments['seconds']} seconds"}
        
        elif tool_name == "task_complete":
            return {
                "success": True,
                "action": "task_complete",
                "summary": arguments.get("summary", "Task completed")
            }
        
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return {"success": False, "error": str(e)}


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time AI chat with browser control"""
    await websocket.accept()
    
    client = await session_manager.get_session(session_id)
    if not client:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return
    
    llama_client = NvidiaLlamaClient(NVIDIA_API_KEY)
    
    # Conversation history
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant controlling a web browser. "
                "You have access to computer tools to interact with the browser: "
                "click_mouse, move_mouse, capture_screenshot, type_text, press_key, scroll, drag_mouse, wait. "
                "ALWAYS start by taking a screenshot to see what's on screen. "
                "Execute actions step by step and describe what you're doing. "
                "When you complete the task successfully, call the task_complete function with a summary. "
                "If the browser opens with a setup wizard, IGNORE IT and proceed with the task. "
                "Be proactive - take screenshots often to verify your actions worked."
            )
        }
    ]
    
    try:
        while True:
            # Wait for user message
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                user_message = data.get("content", "")
                messages.append({"role": "user", "content": user_message})
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "status",
                    "message": "Processing your request..."
                })
                
                # Infinite loop until task_complete is called
                task_completed = False
                max_iterations = 50  # Safety limit
                iteration = 0
                
                while not task_completed and iteration < max_iterations:
                    iteration += 1
                    
                    # Get AI response with streaming
                    full_response = ""
                    tool_calls = []
                    current_tool_call = None
                    
                    try:
                        async for chunk in llama_client.create_chat_completion(
                            messages=messages,
                            tools=COMPUTER_TOOLS,
                            stream=True
                        ):
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                
                                # Handle text content
                                if "content" in delta and delta["content"]:
                                    content = delta["content"]
                                    full_response += content
                                    await websocket.send_json({
                                        "type": "text",
                                        "content": content
                                    })
                                
                                # Handle tool calls
                                if "tool_calls" in delta:
                                    for tc in delta["tool_calls"]:
                                        idx = tc.get("index", 0)
                                        if idx >= len(tool_calls):
                                            tool_calls.append({
                                                "id": tc.get("id", ""),
                                                "function": {"name": "", "arguments": ""}
                                            })
                                        
                                        if "id" in tc and tc["id"]:
                                            tool_calls[idx]["id"] = tc["id"]
                                        
                                        if "function" in tc:
                                            if "name" in tc["function"]:
                                                tool_calls[idx]["function"]["name"] = tc["function"]["name"]
                                            if "arguments" in tc["function"]:
                                                tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
                    
                    except Exception as e:
                        logger.error(f"Error in AI response: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"AI error: {str(e)}"
                        })
                        break
                    
                    # Add assistant response to messages
                    if full_response or tool_calls:
                        assistant_message = {"role": "assistant"}
                        if full_response:
                            assistant_message["content"] = full_response
                        if tool_calls:
                            assistant_message["tool_calls"] = tool_calls
                        messages.append(assistant_message)
                    
                    # Execute tool calls
                    if tool_calls:
                        for tc in tool_calls:
                            tool_name = tc["function"]["name"]
                            try:
                                arguments = json.loads(tc["function"]["arguments"])
                            except json.JSONDecodeError:
                                arguments = {}
                            
                            # Send tool execution notification
                            await websocket.send_json({
                                "type": "tool_call",
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "status": "executing"
                            })
                            
                            # Execute the tool
                            result = await execute_tool(client, tool_name, arguments)
                            
                            # Send tool result
                            await websocket.send_json({
                                "type": "tool_result",
                                "tool_name": tool_name,
                                "result": result
                            })
                            
                            # Add tool result to messages
                            tool_result_content = json.dumps(result) if "type" not in result or result.get("type") != "image" else "Screenshot taken successfully"
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": tool_result_content
                            })
                            
                            # Check if task is complete
                            if tool_name == "task_complete":
                                task_completed = True
                                await websocket.send_json({
                                    "type": "task_complete",
                                    "summary": result.get("summary", "Task completed")
                                })
                                break
                    else:
                        # No tool calls - break the loop
                        if not tool_calls and full_response:
                            break
                
                if not task_completed and iteration >= max_iterations:
                    await websocket.send_json({
                        "type": "warning",
                        "message": "Reached maximum iterations. Task may not be complete."
                    })
            
            elif data.get("type") == "stop":
                await websocket.send_json({
                    "type": "status",
                    "message": "Task stopped by user"
                })
                break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except Exception:
            pass
    finally:
        await llama_client.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
