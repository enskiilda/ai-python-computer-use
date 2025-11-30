# AI Computer Use with Kernel SDK and NVIDIA Llama 4

A real-time AI agent that controls a browser using computer tools. Built with:

- **Kernel SDK** - Browser control (mouse, keyboard, screenshots)
- **NVIDIA Llama 4** - AI model for decision making
- **WebSocket** - Real-time bidirectional communication (not SSE)
- **Next.js** - Frontend UI
- **Python FastAPI** - Backend server

## Architecture

```
┌─────────────┐    WebSocket    ┌──────────────────┐    HTTP    ┌────────────┐
│   Next.js   │ ◄─────────────► │  Python Backend  │ ◄────────► │ Kernel SDK │
│   Frontend  │                 │  (FastAPI)       │            │  Browser   │
└─────────────┘                 │                  │            └────────────┘
                                │  NVIDIA Llama 4  │
                                │  Function Calling│
                                └──────────────────┘
```

## Features

- **Real-time streaming** - AI actions and messages stream live via WebSocket
- **Infinite loop execution** - AI continues executing until task is complete
- **Computer tools** - Click, type, scroll, screenshot, press keys, drag
- **Task completion** - AI signals when task is done with summary
- **Live action updates** - AI reports what it's doing in real-time

## Getting Started

### 1. Start the Python Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The backend runs on `http://localhost:8000`

### 2. Start the Frontend

```bash
pnpm install
pnpm dev
```

The frontend runs on `http://localhost:3000`

## API Keys (Hardcoded in backend/main.py)

The following API keys are hardcoded in the backend:

- **NVIDIA_API_KEY** - For Llama 4 API
- **KERNEL_API_KEY** - For Kernel SDK browser control

## Computer Tools

The AI has access to these tools for controlling the browser:

| Tool | Description |
|------|-------------|
| `click_mouse` | Click at coordinates (x, y) |
| `move_mouse` | Move cursor to position |
| `capture_screenshot` | Take screenshot of browser |
| `type_text` | Type text into focused element |
| `press_key` | Press keyboard keys (Enter, Tab, Ctrl+C, etc.) |
| `scroll` | Scroll page at position |
| `drag_mouse` | Drag along a path of points |
| `wait` | Wait for specified duration |
| `task_complete` | Signal task completion with summary |

## WebSocket Protocol

### Messages from Backend to Frontend

```typescript
{ type: "text", content: string }        // AI text response (streaming)
{ type: "tool_call", tool_name: string, arguments: object, status: "executing" }
{ type: "tool_result", tool_name: string, result: object }
{ type: "task_complete", summary: string }
{ type: "status", message: string }
{ type: "warning", message: string }
{ type: "error", message: string }
```

### Messages from Frontend to Backend

```typescript
{ type: "message", content: string }     // User message
{ type: "stop" }                         // Stop execution
```

## How It Works

1. User sends a task via the chat interface
2. AI takes a screenshot to see the current state
3. AI decides on an action and executes it via Kernel SDK
4. AI continues in a loop, taking screenshots and executing actions
5. When the task is complete, AI calls `task_complete` to signal success
6. All actions are streamed in real-time to the frontend
