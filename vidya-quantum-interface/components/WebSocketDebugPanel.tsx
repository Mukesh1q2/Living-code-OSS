"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  useWebSocketConnection, 
  useWebSocketMessages, 
  ConnectionState,
  WebSocketDebugger 
} from '@/lib/websocket';
import { useWebSocketActions } from './WebSocketProvider';

interface DebugInfo {
  connectionInfo: any;
  messageStats: any;
  recentMessages: any[];
}

export default function WebSocketDebugPanel() {
  const [isOpen, setIsOpen] = useState(false);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [testMessage, setTestMessage] = useState('');
  
  const { isConnected, connectionState, connect, disconnect, lastError, stats } = useWebSocketConnection();
  const { messageHistory, clearHistory } = useWebSocketMessages();
  const { sendUserInput, requestSanskritAnalysis, triggerQuantumStateChange } = useWebSocketActions();

  // Update debug info periodically
  useEffect(() => {
    if (!isOpen) return;

    const updateDebugInfo = () => {
      const connectionInfo = WebSocketDebugger.getConnectionInfo();
      const messageStats = WebSocketDebugger.getMessageStats();
      const recentMessages = messageHistory.slice(-10); // Last 10 messages

      setDebugInfo({
        connectionInfo,
        messageStats,
        recentMessages,
      });
    };

    updateDebugInfo();
    const interval = setInterval(updateDebugInfo, 1000);

    return () => clearInterval(interval);
  }, [isOpen, messageHistory]);

  // Only show in development
  if (process.env.NODE_ENV !== 'development') {
    return null;
  }

  const getConnectionColor = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return '#4ade80';
      case ConnectionState.CONNECTING:
      case ConnectionState.RECONNECTING:
        return '#fbbf24';
      case ConnectionState.ERROR:
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const handleTestMessage = () => {
    if (!testMessage.trim()) return;
    
    WebSocketDebugger.sendTestMessage('debug_test', { message: testMessage });
    setTestMessage('');
  };

  const handleQuickTest = (type: string) => {
    switch (type) {
      case 'ping':
        WebSocketDebugger.sendTestMessage('ping', {});
        break;
      case 'sanskrit':
        requestSanskritAnalysis('नमस्ते');
        break;
      case 'quantum':
        triggerQuantumStateChange('superposition');
        break;
      case 'user_input':
        sendUserInput('Hello Vidya!');
        break;
    }
  };

  return (
    <>
      {/* Debug Toggle Button */}
      <motion.button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: 'fixed',
          bottom: 20,
          left: 20,
          zIndex: 1000,
          width: 50,
          height: 50,
          borderRadius: '50%',
          background: 'rgba(0, 0, 0, 0.8)',
          border: `2px solid ${getConnectionColor()}`,
          color: '#fff',
          fontSize: '16px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        title="WebSocket Debug Panel"
      >
        🔧
      </motion.button>

      {/* Debug Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, x: -20, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: -20, scale: 0.95 }}
            style={{
              position: 'fixed',
              bottom: 80,
              left: 20,
              width: 400,
              maxHeight: '70vh',
              zIndex: 999,
              background: 'rgba(0, 0, 0, 0.95)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: 12,
              backdropFilter: 'blur(20px)',
              overflow: 'hidden',
              fontFamily: 'monospace',
              fontSize: '12px',
            }}
          >
            {/* Header */}
            <div
              style={{
                padding: '12px 16px',
                borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                background: 'rgba(255, 255, 255, 0.05)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <h3 style={{ margin: 0, color: '#fff', fontSize: '14px' }}>
                WebSocket Debug
              </h3>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  color: getConnectionColor(),
                }}
              >
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: getConnectionColor(),
                  }}
                />
                {connectionState}
              </div>
            </div>

            {/* Content */}
            <div
              style={{
                padding: '16px',
                overflowY: 'auto',
                maxHeight: 'calc(70vh - 60px)',
                color: '#fff',
              }}
            >
              {/* Connection Controls */}
              <div style={{ marginBottom: 16 }}>
                <h4 style={{ margin: '0 0 8px 0', color: '#7BE1FF' }}>Connection</h4>
                <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                  <button
                    onClick={connect}
                    disabled={isConnected}
                    style={{
                      padding: '4px 8px',
                      background: isConnected ? '#374151' : '#4ade80',
                      border: 'none',
                      borderRadius: 4,
                      color: '#fff',
                      fontSize: '10px',
                      cursor: isConnected ? 'not-allowed' : 'pointer',
                    }}
                  >
                    Connect
                  </button>
                  <button
                    onClick={disconnect}
                    disabled={!isConnected}
                    style={{
                      padding: '4px 8px',
                      background: !isConnected ? '#374151' : '#ef4444',
                      border: 'none',
                      borderRadius: 4,
                      color: '#fff',
                      fontSize: '10px',
                      cursor: !isConnected ? 'not-allowed' : 'pointer',
                    }}
                  >
                    Disconnect
                  </button>
                </div>
                {lastError && (
                  <div style={{ color: '#ef4444', fontSize: '10px' }}>
                    Error: {lastError}
                  </div>
                )}
              </div>

              {/* Connection Stats */}
              {debugInfo && (
                <div style={{ marginBottom: 16 }}>
                  <h4 style={{ margin: '0 0 8px 0', color: '#7BE1FF' }}>Stats</h4>
                  <div style={{ fontSize: '10px', lineHeight: 1.4 }}>
                    <div>Total Connections: {stats.totalConnections}</div>
                    <div>Total Messages: {stats.totalMessages}</div>
                    <div>Total Errors: {stats.totalErrors}</div>
                    <div>Queue Length: {debugInfo.connectionInfo.queueLength}</div>
                    <div>History Length: {debugInfo.connectionInfo.historyLength}</div>
                  </div>
                </div>
              )}

              {/* Quick Tests */}
              <div style={{ marginBottom: 16 }}>
                <h4 style={{ margin: '0 0 8px 0', color: '#7BE1FF' }}>Quick Tests</h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {['ping', 'sanskrit', 'quantum', 'user_input'].map(type => (
                    <button
                      key={type}
                      onClick={() => handleQuickTest(type)}
                      disabled={!isConnected}
                      style={{
                        padding: '4px 8px',
                        background: isConnected ? '#B383FF' : '#374151',
                        border: 'none',
                        borderRadius: 4,
                        color: '#fff',
                        fontSize: '10px',
                        cursor: isConnected ? 'pointer' : 'not-allowed',
                      }}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              {/* Custom Test Message */}
              <div style={{ marginBottom: 16 }}>
                <h4 style={{ margin: '0 0 8px 0', color: '#7BE1FF' }}>Custom Test</h4>
                <div style={{ display: 'flex', gap: 4 }}>
                  <input
                    type="text"
                    value={testMessage}
                    onChange={(e) => setTestMessage(e.target.value)}
                    placeholder="Test message..."
                    style={{
                      flex: 1,
                      padding: '4px 8px',
                      background: 'rgba(255, 255, 255, 0.1)',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      borderRadius: 4,
                      color: '#fff',
                      fontSize: '10px',
                      outline: 'none',
                    }}
                  />
                  <button
                    onClick={handleTestMessage}
                    disabled={!isConnected || !testMessage.trim()}
                    style={{
                      padding: '4px 8px',
                      background: isConnected && testMessage.trim() ? '#7BE1FF' : '#374151',
                      border: 'none',
                      borderRadius: 4,
                      color: '#000',
                      fontSize: '10px',
                      cursor: isConnected && testMessage.trim() ? 'pointer' : 'not-allowed',
                    }}
                  >
                    Send
                  </button>
                </div>
              </div>

              {/* Message Types */}
              {debugInfo && debugInfo.messageStats.messageTypes && (
                <div style={{ marginBottom: 16 }}>
                  <h4 style={{ margin: '0 0 8px 0', color: '#7BE1FF' }}>Message Types</h4>
                  <div style={{ fontSize: '10px', lineHeight: 1.4 }}>
                    {Object.entries(debugInfo.messageStats.messageTypes).map(([type, count]) => (
                      <div key={type}>
                        {type}: {count as number}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recent Messages */}
              {debugInfo && debugInfo.recentMessages.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <h4 style={{ margin: 0, color: '#7BE1FF' }}>Recent Messages</h4>
                    <button
                      onClick={clearHistory}
                      style={{
                        padding: '2px 6px',
                        background: '#ef4444',
                        border: 'none',
                        borderRadius: 3,
                        color: '#fff',
                        fontSize: '9px',
                        cursor: 'pointer',
                      }}
                    >
                      Clear
                    </button>
                  </div>
                  <div
                    style={{
                      maxHeight: 150,
                      overflowY: 'auto',
                      background: 'rgba(255, 255, 255, 0.05)',
                      borderRadius: 4,
                      padding: 8,
                    }}
                  >
                    {debugInfo.recentMessages.map((msg, index) => (
                      <div
                        key={index}
                        style={{
                          fontSize: '9px',
                          lineHeight: 1.3,
                          marginBottom: 4,
                          paddingBottom: 4,
                          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                        }}
                      >
                        <div style={{ color: '#7BE1FF' }}>
                          {msg.type} ({new Date(msg.timestamp).toLocaleTimeString()})
                        </div>
                        <div style={{ color: '#ccc', marginTop: 2 }}>
                          {JSON.stringify(msg.payload).substring(0, 100)}
                          {JSON.stringify(msg.payload).length > 100 ? '...' : ''}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}