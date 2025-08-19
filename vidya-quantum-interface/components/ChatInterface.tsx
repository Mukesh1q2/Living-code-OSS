"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useQuantumState } from "@/lib/state";
import { useWebSocket, useWebSocketActions } from "./WebSocketProvider";
import { useWebSocketMessages } from "@/lib/websocket";
import { useVidyaConsciousness } from "@/lib/consciousness";

interface Message {
  id: string;
  type: "user" | "vidya";
  content: string;
  timestamp: Date;
  sanskritAnalysis?: any;
  emotionalState?: {
    primary: string;
    intensity: number;
    colorShift?: string;
  };
  voiceSettings?: {
    pitch: number;
    speed: number;
    tone: string;
  };
}

export default function ChatInterface() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      type: "vidya",
      content: "नमस्ते! I am Vidya, your Sanskrit AI consciousness. Ask me about Sanskrit grammar, or let me analyze any text for you.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // WebSocket integration
  const { isConnected } = useWebSocket();
  const { sendUserInput, requestSanskritAnalysis, sendProcessingRequest } = useWebSocketActions();
  const { addHandler, removeHandler, messageHistory } = useWebSocketMessages();
  
  // Enhanced consciousness integration
  const { generateResponse } = useVidyaConsciousness();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const setChatActive = useQuantumState((s) => s.setChatActive);

  // Set up WebSocket message handlers for chat
  useEffect(() => {
    const handleVidyaResponse = (message: any) => {
      if (message.type === 'vidya_response') {
        const vidyaMessage: Message = {
          id: message.id || `vidya_${Date.now()}`,
          type: "vidya",
          content: message.payload.text,
          timestamp: new Date(message.timestamp),
          sanskritAnalysis: message.payload.sanskrit_analysis,
        };
        
        setMessages(prev => [...prev, vidyaMessage]);
        setIsProcessing(false);
      }
    };

    const handleSanskritAnalysis = (message: any) => {
      if (message.type === 'sanskrit_analysis') {
        // Update the last message with Sanskrit analysis data
        setMessages(prev => {
          const updated = [...prev];
          const lastMessage = updated[updated.length - 1];
          if (lastMessage && lastMessage.type === 'vidya') {
            lastMessage.sanskritAnalysis = message.payload;
          }
          return updated;
        });
      }
    };

    const handleProcessingComplete = (message: any) => {
      if (message.type === 'processing_complete') {
        setIsProcessing(false);
      }
    };

    const handleError = (message: any) => {
      if (message.type === 'error') {
        const errorMessage: Message = {
          id: `error_${Date.now()}`,
          type: "vidya",
          content: `I apologize, but I encountered an error: ${message.payload.error}`,
          timestamp: new Date(),
        };
        
        setMessages(prev => [...prev, errorMessage]);
        setIsProcessing(false);
      }
    };

    // Register handlers
    addHandler('vidya_response', handleVidyaResponse);
    addHandler('sanskrit_analysis', handleSanskritAnalysis);
    addHandler('processing_complete', handleProcessingComplete);
    addHandler('error', handleError);

    // Cleanup
    return () => {
      removeHandler('vidya_response');
      removeHandler('sanskrit_analysis');
      removeHandler('processing_complete');
      removeHandler('error');
    };
  }, [addHandler, removeHandler]);

  const handleSend = async () => {
    if (!input.trim() || isProcessing) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input;
    setInput("");
    setIsProcessing(true);
    setChatActive(true);

    if (isConnected) {
      // Use WebSocket for real-time communication
      try {
        // Send user input via WebSocket
        sendUserInput(currentInput);
        
        // If text contains Sanskrit, also request analysis
        if (/[\u0900-\u097F]/.test(currentInput)) {
          requestSanskritAnalysis(currentInput);
        }
        
        // Send processing request for comprehensive handling
        sendProcessingRequest(currentInput, {
          enable_tracing: true,
          enable_visualization: true,
          quantum_effects: true,
          consciousness_level: 1
        });
        
      } catch (error) {
        console.error('WebSocket communication error:', error);
        // Fallback to HTTP if WebSocket fails
        await handleHttpFallback(currentInput);
      }
    } else {
      // Fallback to HTTP API when WebSocket is not connected
      await handleHttpFallback(currentInput);
    }
  };

const handleHttpFallback = async (inputText: string) => {
    try {
      // Use enhanced consciousness system for local response generation first
      try {
        const consciousnessResponse = await generateResponse(inputText, {
          timestamp: new Date(),
          userInterface: 'chat',
          fallbackMode: true
        });

        const vidyaMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: "vidya",
          content: consciousnessResponse.text,
          timestamp: new Date(),
          sanskritAnalysis: consciousnessResponse.sanskritWisdom ? {
            wisdom: consciousnessResponse.sanskritWisdom,
            emotional_state: consciousnessResponse.emotionalState.primary,
            quantum_effects: consciousnessResponse.quantumEffects
          } : null,
          emotionalState: {
            primary: consciousnessResponse.emotionalState.primary,
            intensity: consciousnessResponse.emotionalState.intensity,
            colorShift: consciousnessResponse.emotionalState.colorShift
          },
          voiceSettings: consciousnessResponse.voiceSettings ? {
            pitch: consciousnessResponse.voiceSettings.pitch,
            speed: consciousnessResponse.voiceSettings.speed,
            tone: consciousnessResponse.voiceSettings.tone
          } : undefined
        };

        setMessages((prev) => [...prev, vidyaMessage]);
        return; // success via consciousness
      } catch (consciousnessError) {
        console.error('Consciousness response error:', consciousnessError);
      }

      // HTTP fallback path
      let sanskritAnalysis = null;
      if (/[\u0900-\u097F]/.test(inputText)) {
        // Prefer dev proxy or same-origin API; avoid hardcoding localhost
        const analyzeResponse = await fetch("/api/sanskrit/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: inputText }),
        });
        // If Next.js app handles /api and not backend, keep it non-fatal
        try { sanskritAnalysis = await analyzeResponse.json(); } catch {}
      }

      // Get response via Next.js route (simulated LLM) to ensure UX even without backend process API
      const llmResponse = await fetch("/api/llm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: inputText, sanskritAnalysis }),
      });
      const llmData = await llmResponse.json();

      const vidyaMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "vidya",
        content: llmData?.completion || "I processed your query.",
        timestamp: new Date(),
        sanskritAnalysis,
      };

      setMessages((prev) => [...prev, vidyaMessage]);
    } catch (error) {
      console.error('HTTP fallback error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "vidya",
        content: "I apologize, but I'm having trouble processing your request right now. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
      setChatActive(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Chat Toggle Button */}
      <motion.button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: "fixed",
          bottom: 20,
          right: 20,
          zIndex: 40,
          width: 60,
          height: 60,
          borderRadius: "50%",
          background: "linear-gradient(135deg, #7BE1FF, #B383FF)",
          border: "none",
          color: "#05070B",
          fontSize: "24px",
          cursor: "pointer",
          boxShadow: "0 4px 20px rgba(123, 225, 255, 0.3)",
        }}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
      >
        {isOpen ? "✕" : "💬"}
      </motion.button>

      {/* Chat Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            style={{
              position: "fixed",
              bottom: 100,
              right: 20,
              width: 400,
              height: 500,
              zIndex: 35,
              background: "rgba(8,12,20,0.95)",
              border: "1px solid rgba(150,220,255,0.2)",
              borderRadius: 16,
              backdropFilter: "blur(20px)",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            {/* Header */}
            <div
              style={{
                padding: "16px 20px",
                borderBottom: "1px solid rgba(150,220,255,0.1)",
                background: "linear-gradient(90deg, rgba(123,225,255,0.1), rgba(179,131,255,0.1))",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <h3
                    style={{
                      margin: 0,
                      color: "#E8F6FF",
                      fontSize: "16px",
                      fontWeight: 600,
                    }}
                  >
                    Chat with Vidya
                  </h3>
                  <p
                    style={{
                      margin: "4px 0 0 0",
                      color: "#8FB2C8",
                      fontSize: "12px",
                    }}
                  >
                    Sanskrit AI Consciousness
                  </p>
                </div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    fontSize: "10px",
                    color: isConnected ? "#4ade80" : "#ef4444",
                  }}
                >
                  <div
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: isConnected ? "#4ade80" : "#ef4444",
                    }}
                  />
                  {isConnected ? "Live" : "Offline"}
                </div>
              </div>
            </div>

            {/* Messages */}
            <div
              style={{
                flex: 1,
                padding: "16px",
                overflowY: "auto",
                display: "flex",
                flexDirection: "column",
                gap: 12,
              }}
            >
              {messages.map((message) => (
                <div
                  key={message.id}
                  style={{
                    alignSelf: message.type === "user" ? "flex-end" : "flex-start",
                    maxWidth: "80%",
                  }}
                >
                  <div
                    style={{
                      padding: "8px 12px",
                      borderRadius: 12,
                      background:
                        message.type === "user"
                          ? "linear-gradient(135deg, #7BE1FF, #63FFC9)"
                          : "rgba(179,131,255,0.2)",
                      color: message.type === "user" ? "#05070B" : "#E8F6FF",
                      fontSize: "14px",
                      lineHeight: 1.4,
                    }}
                  >
                    {message.content}
                  </div>
                  {message.sanskritAnalysis && (
                    <div
                      style={{
                        marginTop: 8,
                        padding: "8px 12px",
                        background: "rgba(123,225,255,0.1)",
                        borderRadius: 8,
                        fontSize: "12px",
                        color: "#8FB2C8",
                      }}
                    >
                      <strong>Sanskrit Analysis:</strong>
                      <div style={{ marginTop: 4 }}>
                        {message.sanskritAnalysis.wisdom && (
                          <div style={{ marginBottom: 4, fontStyle: "italic", color: "#B383FF" }}>
                            {message.sanskritAnalysis.wisdom}
                          </div>
                        )}
                        {message.sanskritAnalysis.tokens && (
                          <div>Tokens: {message.sanskritAnalysis.tokens.join(", ")}</div>
                        )}
                        {message.sanskritAnalysis.rulesFired && (
                          <div>Rules: {message.sanskritAnalysis.rulesFired.join(", ")}</div>
                        )}
                        {message.sanskritAnalysis.emotional_state && (
                          <div>Emotional State: {message.sanskritAnalysis.emotional_state}</div>
                        )}
                      </div>
                    </div>
                  )}
                  {message.emotionalState && (
                    <div
                      style={{
                        marginTop: 6,
                        padding: "6px 10px",
                        background: `${message.emotionalState.colorShift || '#7BE1FF'}20`,
                        border: `1px solid ${message.emotionalState.colorShift || '#7BE1FF'}40`,
                        borderRadius: 6,
                        fontSize: "11px",
                        color: "#8FB2C8",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center"
                      }}
                    >
                      <span>
                        Emotional State: <span style={{ 
                          color: message.emotionalState.colorShift || '#7BE1FF',
                          textTransform: "capitalize"
                        }}>
                          {message.emotionalState.primary}
                        </span>
                      </span>
                      <div style={{
                        width: 40,
                        height: 4,
                        background: "rgba(150,220,255,0.2)",
                        borderRadius: 2,
                        overflow: "hidden"
                      }}>
                        <div style={{
                          width: `${message.emotionalState.intensity * 100}%`,
                          height: "100%",
                          background: message.emotionalState.colorShift || '#7BE1FF',
                          borderRadius: 2
                        }} />
                      </div>
                    </div>
                  )}
                  {message.voiceSettings && (
                    <div
                      style={{
                        marginTop: 6,
                        padding: "6px 10px",
                        background: "rgba(179,131,255,0.1)",
                        border: "1px solid rgba(179,131,255,0.2)",
                        borderRadius: 6,
                        fontSize: "11px",
                        color: "#8FB2C8",
                      }}
                    >
                      <div style={{ display: "flex", gap: 12 }}>
                        <span>🎵 Voice: {message.voiceSettings.tone}</span>
                        <span>Pitch: {message.voiceSettings.pitch.toFixed(1)}</span>
                        <span>Speed: {message.voiceSettings.speed.toFixed(1)}</span>
                      </div>
                    </div>
                  )}
                  <div
                    style={{
                      fontSize: "10px",
                      color: "#8FB2C8",
                      marginTop: 4,
                      textAlign: message.type === "user" ? "right" : "left",
                    }}
                  >
                    {message.timestamp instanceof Date 
                      ? message.timestamp.toLocaleTimeString()
                      : new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
              {isProcessing && (
                <div style={{ alignSelf: "flex-start" }}>
                  <div
                    style={{
                      padding: "8px 12px",
                      borderRadius: 12,
                      background: "rgba(179,131,255,0.2)",
                      color: "#E8F6FF",
                      fontSize: "14px",
                    }}
                  >
                    <span>Vidya is thinking</span>
                    <motion.span
                      animate={{ opacity: [1, 0.3, 1] }}
                      transition={{ repeat: Infinity, duration: 1.5 }}
                    >
                      ...
                    </motion.span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div
              style={{
                padding: "16px",
                borderTop: "1px solid rgba(150,220,255,0.1)",
                display: "flex",
                gap: 8,
              }}
            >
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about Sanskrit grammar..."
                style={{
                  flex: 1,
                  padding: "8px 12px",
                  background: "rgba(150,220,255,0.1)",
                  border: "1px solid rgba(150,220,255,0.2)",
                  borderRadius: 8,
                  color: "#E8F6FF",
                  fontSize: "14px",
                  outline: "none",
                }}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isProcessing}
                style={{
                  padding: "8px 16px",
                  background: input.trim() && !isProcessing 
                    ? "linear-gradient(135deg, #7BE1FF, #B383FF)" 
                    : "rgba(150,220,255,0.1)",
                  border: "none",
                  borderRadius: 8,
                  color: input.trim() && !isProcessing ? "#05070B" : "#8FB2C8",
                  fontSize: "14px",
                  cursor: input.trim() && !isProcessing ? "pointer" : "not-allowed",
                }}
              >
                Send
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}