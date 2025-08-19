import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useWebSocket } from './WebSocketContext';
import { apiService } from '../services/apiService';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

interface ChatContextType {
  conversations: Conversation[];
  currentConversation: Conversation | null;
  isLoading: boolean;
  error: string | null;
  createConversation: (title?: string) => Conversation;
  selectConversation: (conversationId: string) => void;
  sendMessage: (content: string, options?: { stream?: boolean }) => Promise<void>;
  deleteConversation: (conversationId: string) => void;
  clearError: () => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};

interface ChatProviderProps {
  children: React.ReactNode;
}

export const ChatProvider: React.FC<ChatProviderProps> = ({ children }) => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { sendMessage: sendWebSocketMessage, onMessage, offMessage } = useWebSocket();

  // Handle WebSocket messages
  useEffect(() => {
    const handleMessage = (type: string, data: any) => {
      if (type === 'chat_response') {
        const message: ChatMessage = {
          id: data.message.id,
          role: data.message.role,
          content: data.message.content,
          timestamp: new Date(data.message.timestamp),
          metadata: data.message.metadata,
        };

        setConversations(prev => 
          prev.map(conv => 
            conv.id === data.conversation_id
              ? {
                  ...conv,
                  messages: [...conv.messages, message],
                  updatedAt: new Date(),
                }
              : conv
          )
        );

        if (currentConversation?.id === data.conversation_id) {
          setCurrentConversation(prev => 
            prev ? {
              ...prev,
              messages: [...prev.messages, message],
              updatedAt: new Date(),
            } : null
          );
        }

        setIsLoading(false);
      }
    };

    onMessage(handleMessage);
    return () => offMessage(handleMessage);
  }, [onMessage, offMessage, currentConversation?.id]);

  const createConversation = useCallback((title?: string): Conversation => {
    const newConversation: Conversation = {
      id: uuidv4(),
      title: title || 'New Conversation',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    setConversations(prev => [newConversation, ...prev]);
    setCurrentConversation(newConversation);
    setError(null);

    return newConversation;
  }, []);

  const selectConversation = useCallback((conversationId: string) => {
    const conversation = conversations.find(conv => conv.id === conversationId);
    if (conversation) {
      setCurrentConversation(conversation);
      setError(null);
    }
  }, [conversations]);

  const sendMessage = useCallback(async (content: string, options?: { stream?: boolean }) => {
    if (!currentConversation) {
      setError('No conversation selected');
      return;
    }

    setIsLoading(true);
    setError(null);

    // Add user message immediately
    const userMessage: ChatMessage = {
      id: uuidv4(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    const updatedConversation = {
      ...currentConversation,
      messages: [...currentConversation.messages, userMessage],
      updatedAt: new Date(),
    };

    setCurrentConversation(updatedConversation);
    setConversations(prev => 
      prev.map(conv => 
        conv.id === currentConversation.id ? updatedConversation : conv
      )
    );

    try {
      if (options?.stream) {
        // Use WebSocket for streaming
        sendWebSocketMessage('chat', {
          message: content,
          conversation_id: currentConversation.id,
          stream: true,
        });
      } else {
        // Use REST API for non-streaming
        const response = await apiService.sendChatMessage({
          message: content,
          conversation_id: currentConversation.id,
          stream: false,
        });

        const assistantMessage: ChatMessage = {
          id: response.message.id,
          role: response.message.role as 'user' | 'assistant' | 'system',
          content: response.message.content,
          timestamp: new Date(response.message.timestamp),
          metadata: response.message.metadata,
        };

        const finalConversation = {
          ...updatedConversation,
          messages: [...updatedConversation.messages, assistantMessage],
          updatedAt: new Date(),
        };

        setCurrentConversation(finalConversation);
        setConversations(prev => 
          prev.map(conv => 
            conv.id === currentConversation.id ? finalConversation : conv
          )
        );

        setIsLoading(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      setIsLoading(false);
    }
  }, [currentConversation, sendWebSocketMessage]);

  const deleteConversation = useCallback((conversationId: string) => {
    setConversations(prev => prev.filter(conv => conv.id !== conversationId));
    
    if (currentConversation?.id === conversationId) {
      const remainingConversations = conversations.filter(conv => conv.id !== conversationId);
      setCurrentConversation(remainingConversations.length > 0 ? remainingConversations[0] : null);
    }
  }, [conversations, currentConversation?.id]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Initialize with a default conversation
  useEffect(() => {
    if (conversations.length === 0) {
      createConversation('Welcome to Sanskrit Engine');
    }
  }, [conversations.length, createConversation]);

  const value: ChatContextType = {
    conversations,
    currentConversation,
    isLoading,
    error,
    createConversation,
    selectConversation,
    sendMessage,
    deleteConversation,
    clearError,
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
};