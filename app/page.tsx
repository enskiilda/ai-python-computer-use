"use client";

import { PreviewMessage } from "@/components/message";
import { useScrollToBottom } from "@/lib/use-scroll-to-bottom";
import { useEffect, useState, useCallback, useRef } from "react";
import { Input } from "@/components/input";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { DeployButton, ProjectInfo } from "@/components/project-info";
import { AISDKLogo } from "@/components/icons";
import { PromptSuggestions } from "@/components/prompt-suggestions";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import type { Message } from "ai";

// Backend URL - hardcoded for unsafe mode
const BACKEND_URL = "http://localhost:8000";

// Message types for the chat
interface ChatMessage extends Message {
  toolCalls?: Array<{
    toolName: string;
    arguments: Record<string, unknown>;
    status: "executing" | "completed";
    result?: Record<string, unknown>;
  }>;
}

export default function Chat() {
  // Create separate refs for mobile and desktop to ensure both scroll properly
  const [desktopContainerRef, desktopEndRef] = useScrollToBottom();
  const [mobileContainerRef, mobileEndRef] = useScrollToBottom();

  const [isInitializing, setIsInitializing] = useState(true);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<"ready" | "streaming" | "submitted" | "error">("ready");
  const wsRef = useRef<WebSocket | null>(null);
  const currentMessageRef = useRef<string>("");

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInput(event.target.value);
  };

  // Create session and WebSocket connection
  const initSession = useCallback(async () => {
    try {
      setIsInitializing(true);
      
      // Create new browser session via Python backend
      const response = await fetch(`${BACKEND_URL}/api/session/create`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to create session: ${response.statusText}`);
      }
      
      const data = await response.json();
      setSessionId(data.session_id);
      setStreamUrl(data.stream_url);
      
      return data.session_id;
    } catch (error) {
      console.error("Failed to initialize session:", error);
      toast.error("Failed to initialize browser session");
      return null;
    } finally {
      setIsInitializing(false);
    }
  }, []);

  // Connect WebSocket
  const connectWebSocket = useCallback((sid: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }
    
    const ws = new WebSocket(`ws://localhost:8000/ws/chat/${sid}`);
    
    ws.onopen = () => {
      console.log("WebSocket connected");
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case "text":
          // Streaming text from AI
          currentMessageRef.current += data.content;
          setMessages((prev) => {
            const lastMsg = prev[prev.length - 1];
            if (lastMsg?.role === "assistant" && !lastMsg.toolCalls?.length) {
              return [
                ...prev.slice(0, -1),
                {
                  ...lastMsg,
                  content: currentMessageRef.current,
                  parts: [{ type: "text" as const, text: currentMessageRef.current }],
                },
              ];
            } else {
              return [
                ...prev,
                {
                  id: `msg-${Date.now()}`,
                  role: "assistant" as const,
                  content: currentMessageRef.current,
                  parts: [{ type: "text" as const, text: currentMessageRef.current }],
                },
              ];
            }
          });
          break;
        
        case "tool_call":
          // AI is calling a tool
          currentMessageRef.current = "";
          setMessages((prev) => {
            const toolInvocation = {
              toolCallId: `tool-${Date.now()}`,
              toolName: data.tool_name,
              args: data.arguments,
              state: "call" as const,
            };
            
            const lastMsg = prev[prev.length - 1];
            if (lastMsg?.role === "assistant") {
              return [
                ...prev.slice(0, -1),
                {
                  ...lastMsg,
                  parts: [
                    ...(lastMsg.parts || []),
                    { type: "tool-invocation" as const, toolInvocation },
                  ],
                },
              ];
            } else {
              return [
                ...prev,
                {
                  id: `msg-${Date.now()}`,
                  role: "assistant" as const,
                  content: "",
                  parts: [{ type: "tool-invocation" as const, toolInvocation }],
                },
              ];
            }
          });
          break;
        
        case "tool_result":
          // Tool execution completed
          setMessages((prev) => {
            const lastMsg = prev[prev.length - 1];
            if (lastMsg?.role === "assistant" && lastMsg.parts) {
              const updatedParts = lastMsg.parts.map((part) => {
                if (
                  part.type === "tool-invocation" &&
                  part.toolInvocation.toolName === data.tool_name &&
                  part.toolInvocation.state === "call"
                ) {
                  return {
                    ...part,
                    toolInvocation: {
                      ...part.toolInvocation,
                      state: "result" as const,
                      result: data.result,
                    },
                  };
                }
                return part;
              });
              
              return [
                ...prev.slice(0, -1),
                { ...lastMsg, parts: updatedParts },
              ];
            }
            return prev;
          });
          break;
        
        case "task_complete":
          setStatus("ready");
          toast.success("Task completed!", {
            description: data.summary,
            position: "top-center",
          });
          break;
        
        case "status":
          console.log("Status:", data.message);
          break;
        
        case "warning":
          toast.warning(data.message, { position: "top-center" });
          setStatus("ready");
          break;
        
        case "error":
          toast.error(data.message, { position: "top-center" });
          setStatus("error");
          break;
      }
    };
    
    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      toast.error("Connection error");
      setStatus("error");
    };
    
    ws.onclose = () => {
      console.log("WebSocket closed");
    };
    
    wsRef.current = ws;
  }, []);

  // Submit message
  const handleSubmit = useCallback((e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (!input.trim() || !sessionId || status !== "ready") return;
    
    // Add user message
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: input,
      parts: [{ type: "text", text: input }],
    };
    
    setMessages((prev) => [...prev, userMessage]);
    currentMessageRef.current = "";
    setStatus("streaming");
    
    // Send via WebSocket
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: "message",
        content: input,
      }));
    } else {
      // Reconnect and send
      connectWebSocket(sessionId);
      setTimeout(() => {
        wsRef.current?.send(JSON.stringify({
          type: "message",
          content: input,
        }));
      }, 500);
    }
    
    setInput("");
  }, [input, sessionId, status, connectWebSocket]);

  // Stop generation
  const stop = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
    }
    setStatus("ready");
  }, []);

  // Append message (for suggestions)
  const append = useCallback((message: { role: string; content: string }) => {
    setInput(message.content);
    setTimeout(() => {
      const fakeEvent = { preventDefault: () => {} } as React.FormEvent;
      handleSubmit(fakeEvent);
    }, 100);
  }, [handleSubmit]);

  const isLoading = status !== "ready";

  // Refresh desktop connection
  const refreshDesktop = async () => {
    try {
      // Close existing WebSocket
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      
      // Kill existing session
      if (sessionId) {
        try {
          await fetch(`${BACKEND_URL}/api/session/${sessionId}`, {
            method: "DELETE",
          });
        } catch (e) {
          console.error("Failed to kill old session:", e);
        }
      }
      
      // Create new session
      const newSessionId = await initSession();
      if (newSessionId) {
        connectWebSocket(newSessionId);
        setMessages([]);
      }
    } catch (err) {
      console.error("Failed to refresh desktop:", err);
      toast.error("Failed to refresh desktop");
    }
  };

  // Kill desktop on page close
  useEffect(() => {
    if (!sessionId) return;

    const killDesktop = () => {
      if (!sessionId) return;
      
      // Close WebSocket
      if (wsRef.current) {
        wsRef.current.close();
      }
      
      // Use sendBeacon for cleanup
      navigator.sendBeacon(
        `${BACKEND_URL}/api/session/${encodeURIComponent(sessionId)}`,
      );
    };

    // Detect iOS / Safari
    const isIOS =
      /iPad|iPhone|iPod/.test(navigator.userAgent) ||
      (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

    if (isIOS || isSafari) {
      window.addEventListener("pagehide", killDesktop);
      return () => {
        window.removeEventListener("pagehide", killDesktop);
        killDesktop();
      };
    } else {
      window.addEventListener("beforeunload", killDesktop);
      return () => {
        window.removeEventListener("beforeunload", killDesktop);
        killDesktop();
      };
    }
  }, [sessionId]);

  // Initialize on mount
  useEffect(() => {
    const init = async () => {
      const sid = await initSession();
      if (sid) {
        connectWebSocket(sid);
      }
    };
    
    init();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex h-dvh relative">
      {/* Mobile/tablet banner */}
      <div className="flex items-center justify-center fixed left-1/2 -translate-x-1/2 top-5 shadow-md text-xs mx-auto rounded-lg h-8 w-fit bg-blue-600 text-white px-3 py-2 text-left z-50 xl:hidden">
        <span>Headless mode</span>
      </div>

      {/* Resizable Panels */}
      <div className="w-full hidden xl:block">
        <ResizablePanelGroup direction="horizontal" className="h-full">
          {/* Desktop Stream Panel */}
          <ResizablePanel
            defaultSize={70}
            minSize={40}
            className="bg-black relative items-center justify-center"
          >
            {streamUrl ? (
              <>
                <iframe
                  src={streamUrl}
                  className="w-full h-full"
                  style={{
                    transformOrigin: "center",
                    width: "100%",
                    height: "100%",
                  }}
                  allow="autoplay"
                />
                <Button
                  onClick={refreshDesktop}
                  className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 text-white px-3 py-1 rounded text-sm z-10"
                  disabled={isInitializing}
                >
                  {isInitializing ? "Creating desktop..." : "New desktop"}
                </Button>
              </>
            ) : (
              <div className="flex items-center justify-center h-full text-white">
                {isInitializing
                  ? "Initializing desktop..."
                  : "Loading stream..."}
              </div>
            )}
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Chat Interface Panel */}
          <ResizablePanel
            defaultSize={30}
            minSize={25}
            className="flex flex-col border-l border-zinc-200"
          >
            <div className="bg-white py-4 px-4 flex justify-between items-center">
              <AISDKLogo />
              <DeployButton />
            </div>

            <div
              className="flex-1 space-y-6 py-4 overflow-y-auto px-4"
              ref={desktopContainerRef}
            >
              {messages.length === 0 ? <ProjectInfo /> : null}
              {messages.map((message, i) => (
                <PreviewMessage
                  message={message}
                  key={message.id}
                  isLoading={isLoading}
                  status={status}
                  isLatestMessage={i === messages.length - 1}
                />
              ))}
              <div ref={desktopEndRef} className="pb-2" />
            </div>

            {messages.length === 0 && (
              <PromptSuggestions
                disabled={isInitializing}
                submitPrompt={(prompt: string) =>
                  append({ role: "user", content: prompt })
                }
              />
            )}
            <div className="bg-white">
              <form onSubmit={handleSubmit} className="p-4">
                <Input
                  handleInputChange={handleInputChange}
                  input={input}
                  isInitializing={isInitializing}
                  isLoading={isLoading}
                  status={status}
                  stop={stop}
                />
              </form>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Mobile View (Chat Only) */}
      <div className="w-full xl:hidden flex flex-col">
        <div className="bg-white py-4 px-4 flex justify-between items-center">
          <AISDKLogo />
          <DeployButton />
        </div>

        <div
          className="flex-1 space-y-6 py-4 overflow-y-auto px-4"
          ref={mobileContainerRef}
        >
          {messages.length === 0 ? <ProjectInfo /> : null}
          {messages.map((message, i) => (
            <PreviewMessage
              message={message}
              key={message.id}
              isLoading={isLoading}
              status={status}
              isLatestMessage={i === messages.length - 1}
            />
          ))}
          <div ref={mobileEndRef} className="pb-2" />
        </div>

        {messages.length === 0 && (
          <PromptSuggestions
            disabled={isInitializing}
            submitPrompt={(prompt: string) =>
              append({ role: "user", content: prompt })
            }
          />
        )}
        <div className="bg-white">
          <form onSubmit={handleSubmit} className="p-4">
            <Input
              handleInputChange={handleInputChange}
              input={input}
              isInitializing={isInitializing}
              isLoading={isLoading}
              status={status}
              stop={stop}
            />
          </form>
        </div>
      </div>
    </div>
  );
}
