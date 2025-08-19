"use client";

import React, { createContext, useContext, useEffect, ReactNode } from 'react';
import { 
  useWebSocketStore, 
  useWebSocketConnection, 
  useWebSocketMessages,
  TypedWebSocketMessage,
  ConnectionState 
} from '@/lib/websocket';
import { useQuantumState } from '@/lib/state';

// Context for WebSocket functionality
interface WebSocketContextType {
  isConnected: boolean;
  connectionState: ConnectionState;
  sendMessage: (message: any) => void;
  lastError: string | null;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

// Hook to use WebSocket context
export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: ReactNode;
  autoConnect?: boolean;
  debugMode?: boolean;
}

export default function WebSocketProvider({ 
  children, 
  autoConnect = true,
  debugMode = process.env.NODE_ENV === 'development'
}: WebSocketProviderProps) {
  const { connect, disconnect, isConnected, connectionState, lastError } = useWebSocketConnection({
    url: typeof window !== 'undefined' ? (process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws') : undefined,
  });
  const { sendMessage, addHandler, removeHandler } = useWebSocketMessages();
  
  // Quantum state for integration
  const setSelectedNode = useQuantumState(state => state.setSelectedNode);
  const setEntangledNodes = useQuantumState(state => state.setEntangledNodes);
  const setSuperposition = useQuantumState(state => state.setSuperposition);
  const setChatActive = useQuantumState(state => state.setChatActive);

  // Set debug mode
  useEffect(() => {
    useWebSocketStore.getState().setDebugMode(debugMode);
  }, [debugMode]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && !isConnected && connectionState === ConnectionState.DISCONNECTED) {
      console.log('[WebSocketProvider] Auto-connecting to WebSocket...');
      connect();
    }

    // Cleanup on unmount
    return () => {
      if (isConnected) {
        disconnect();
      }
    };
  }, [autoConnect, connect, disconnect, isConnected, connectionState]);

  // Set up message handlers
  useEffect(() => {
    // Handler for consciousness updates
    const handleConsciousnessUpdate = (message: TypedWebSocketMessage) => {
      if (message.type === 'consciousness_update') {
        console.log('[WebSocketProvider] Consciousness updated:', message.payload);
        // Update quantum state based on consciousness changes
        if (message.payload.quantum_coherence > 0.8) {
          setSuperposition(true);
        }
      }
    };

    // Handler for Sanskrit analysis results
    const handleSanskritAnalysis = (message: TypedWebSocketMessage) => {
      if (message.type === 'sanskrit_analysis') {
        console.log('[WebSocketProvider] Sanskrit analysis received:', message.payload);
        setChatActive(false); // Analysis complete
        
        // Update neural network visualization based on analysis
        if (message.payload.network_nodes && message.payload.network_nodes.length > 0) {
          const nodeIds = message.payload.network_nodes.map((node: any) => node.id);
          setEntangledNodes(nodeIds);
        }
      }
    };

    // Handler for processing updates
    const handleProcessingUpdate = (message: TypedWebSocketMessage) => {
      if (message.type === 'sanskrit_processing_update') {
        console.log('[WebSocketProvider] Processing update:', message.payload);
        setChatActive(true); // Processing in progress
      }
    };

    // Handler for neural network updates
    const handleNeuralNetworkUpdate = (message: TypedWebSocketMessage) => {
      if (message.type === 'neural_network_update') {
        console.log('[WebSocketProvider] Neural network updated:', message.payload);
        // Update visualization state
      }
    };

    // Handler for quantum state changes
    const handleQuantumStateChange = (message: TypedWebSocketMessage) => {
      if (message.type === 'quantum_state_change') {
        console.log('[WebSocketProvider] Quantum state changed:', message.payload);
        
        const change = message.payload.change;
        if (change === 'superposition') {
          setSuperposition(true);
        } else if (change === 'collapse') {
          setSuperposition(false);
        }
      }
    };

    // Handler for Vidya responses
    const handleVidyaResponse = (message: TypedWebSocketMessage) => {
      if (message.type === 'vidya_response') {
        console.log('[WebSocketProvider] Vidya response:', message.payload);
        setChatActive(false); // Response received
      }
    };

    // Handler for errors
    const handleError = (message: TypedWebSocketMessage) => {
      if (message.type === 'error') {
        console.error('[WebSocketProvider] Server error:', message.payload);
        setChatActive(false); // Stop processing on error
      }
    };

    // Handler for connection established
    const handleConnectionEstablished = (message: TypedWebSocketMessage) => {
      if (message.type === 'connection_established') {
        console.log('[WebSocketProvider] Connection established:', message.payload);
      }
    };

    // Handler for processing complete
    const handleProcessingComplete = (message: TypedWebSocketMessage) => {
      if (message.type === 'processing_complete') {
        console.log('[WebSocketProvider] Processing complete:', message.payload);
        setChatActive(false);
      }
    };

    // Handler for consciousness evolution
    const handleConsciousnessEvolved = (message: TypedWebSocketMessage) => {
      if (message.type === 'consciousness_evolved') {
        console.log('[WebSocketProvider] Consciousness evolved:', message.payload);
        // Trigger visual effects for consciousness evolution
        setSuperposition(true);
        setTimeout(() => setSuperposition(false), 2000); // Brief superposition effect
      }
    };

    // Register all handlers
    addHandler('consciousness_update', handleConsciousnessUpdate);
    addHandler('sanskrit_analysis', handleSanskritAnalysis);
    addHandler('sanskrit_processing_update', handleProcessingUpdate);
    addHandler('neural_network_update', handleNeuralNetworkUpdate);
    addHandler('quantum_state_change', handleQuantumStateChange);
    addHandler('vidya_response', handleVidyaResponse);
    addHandler('error', handleError);
    addHandler('connection_established', handleConnectionEstablished);
    addHandler('processing_complete', handleProcessingComplete);
    addHandler('consciousness_evolved', handleConsciousnessEvolved);

    // Cleanup handlers on unmount
    return () => {
      removeHandler('consciousness_update');
      removeHandler('sanskrit_analysis');
      removeHandler('sanskrit_processing_update');
      removeHandler('neural_network_update');
      removeHandler('quantum_state_change');
      removeHandler('vidya_response');
      removeHandler('error');
      removeHandler('connection_established');
      removeHandler('processing_complete');
      removeHandler('consciousness_evolved');
    };
  }, [addHandler, removeHandler, setSelectedNode, setEntangledNodes, setSuperposition, setChatActive]);

  // Send periodic ping to keep connection alive
  useEffect(() => {
    if (!isConnected) return;

    const pingInterval = setInterval(() => {
      sendMessage({
        type: 'ping',
        payload: {}
      });
    }, 30000); // Ping every 30 seconds

    return () => clearInterval(pingInterval);
  }, [isConnected, sendMessage]);

  // Context value
  const contextValue: WebSocketContextType = {
    isConnected,
    connectionState,
    sendMessage,
    lastError,
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
}

// Connection status indicator component
export function WebSocketStatus() {
  const { isConnected, connectionState, lastError } = useWebSocket();

  if (process.env.NODE_ENV !== 'development') {
    return null; // Only show in development
  }

  const getStatusColor = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return '#4ade80'; // green
      case ConnectionState.CONNECTING:
      case ConnectionState.RECONNECTING:
        return '#fbbf24'; // yellow
      case ConnectionState.ERROR:
        return '#ef4444'; // red
      default:
        return '#6b7280'; // gray
    }
  };

  const getStatusText = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return 'Connected';
      case ConnectionState.CONNECTING:
        return 'Connecting...';
      case ConnectionState.RECONNECTING:
        return 'Reconnecting...';
      case ConnectionState.ERROR:
        return 'Error';
      case ConnectionState.DISCONNECTED:
        return 'Disconnected';
      default:
        return 'Unknown';
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 10,
        right: 10,
        zIndex: 1000,
        padding: '8px 12px',
        background: 'rgba(0, 0, 0, 0.8)',
        border: `1px solid ${getStatusColor()}`,
        borderRadius: 6,
        color: '#fff',
        fontSize: '12px',
        fontFamily: 'monospace',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}
    >
      <div
        style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: getStatusColor(),
        }}
      />
      <span>WS: {getStatusText()}</span>
      {lastError && (
        <span style={{ color: '#ef4444', marginLeft: 8 }}>
          ({lastError})
        </span>
      )}
    </div>
  );
}

// Hook for sending specific message types
export const useWebSocketActions = () => {
  const { sendMessage } = useWebSocket();

  return {
    // Send user input to Vidya
    sendUserInput: (input: string) => {
      sendMessage({
        type: 'user_input',
        payload: { input }
      });
    },

    // Request Sanskrit analysis
    requestSanskritAnalysis: (text: string) => {
      sendMessage({
        type: 'analyze_sanskrit',
        payload: { text }
      });
    },

    // Trigger quantum state change
    triggerQuantumStateChange: (change: string) => {
      sendMessage({
        type: 'quantum_state_change',
        payload: { change }
      });
    },

    // Request consciousness evolution
    requestConsciousnessEvolution: (levelIncrease: number = 1) => {
      sendMessage({
        type: 'evolve_consciousness',
        payload: { level_increase: levelIncrease }
      });
    },

    // Send processing request
    sendProcessingRequest: (text: string, options: any = {}) => {
      sendMessage({
        type: 'process_text',
        payload: { 
          text, 
          enable_tracing: true,
          enable_visualization: true,
          quantum_effects: true,
          consciousness_level: 1,
          ...options
        }
      });
    },
  };
};