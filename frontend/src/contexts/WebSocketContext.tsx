import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';

interface WebSocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  clientId: string;
  sendMessage: (type: string, data: any) => void;
  onMessage: (callback: (type: string, data: any) => void) => void;
  offMessage: (callback: (type: string, data: any) => void) => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: React.ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [clientId] = useState(() => uuidv4());
  const [messageCallbacks, setMessageCallbacks] = useState<Set<(type: string, data: any) => void>>(new Set());

  useEffect(() => {
    // Initialize WebSocket connection
    const newSocket = io(`ws://localhost:8000/ws/${clientId}`, {
      transports: ['websocket'],
      upgrade: false,
    });

    newSocket.on('connect', () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    });

    newSocket.on('message', (message: string) => {
      try {
        const parsed = JSON.parse(message);
        messageCallbacks.forEach(callback => {
          callback(parsed.type, parsed.data);
        });
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });

    newSocket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setIsConnected(false);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, [clientId]);

  const sendMessage = useCallback((type: string, data: any) => {
    if (socket && isConnected) {
      socket.emit('message', JSON.stringify({ type, data }));
    } else {
      console.warn('WebSocket not connected, message not sent:', { type, data });
    }
  }, [socket, isConnected]);

  const onMessage = useCallback((callback: (type: string, data: any) => void) => {
    setMessageCallbacks(prev => new Set(prev).add(callback));
  }, []);

  const offMessage = useCallback((callback: (type: string, data: any) => void) => {
    setMessageCallbacks(prev => {
      const newSet = new Set(prev);
      newSet.delete(callback);
      return newSet;
    });
  }, []);

  const value: WebSocketContextType = {
    socket,
    isConnected,
    clientId,
    sendMessage,
    onMessage,
    offMessage,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};