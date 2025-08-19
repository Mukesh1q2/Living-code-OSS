import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../contexts/ChatContext';
import ConversationList from './ConversationList';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import './ChatInterface.css';

const ChatInterface: React.FC = () => {
  const {
    currentConversation,
    isLoading,
    error,
    sendMessage,
    clearError,
  } = useChat();

  const [inputValue, setInputValue] = useState('');
  const [streamingEnabled, setStreamingEnabled] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentConversation?.messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    try {
      await sendMessage(content, { stream: streamingEnabled });
      setInputValue('');
    } catch (err) {
      console.error('Failed to send message:', err);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(inputValue);
    }
  };

  return (
    <div className="chat-interface" data-testid="chat-interface">
      <div className="chat-layout">
        <div className="chat-main">
          <div className="chat-header">
            <h2 className="conversation-title">
              {currentConversation?.title || 'Select a conversation'}
            </h2>
            <div className="chat-controls">
              <label className="streaming-toggle">
                <input
                  type="checkbox"
                  checked={streamingEnabled}
                  onChange={(e) => setStreamingEnabled(e.target.checked)}
                />
                <span>Real-time streaming</span>
              </label>
            </div>
          </div>

          {error && (
            <div className="error-banner">
              <span>{error}</span>
              <button onClick={clearError} className="error-close">
                ×
              </button>
            </div>
          )}

          <div className="messages-container">
            {currentConversation ? (
              <>
                <MessageList
                  messages={currentConversation.messages}
                  isLoading={isLoading}
                />
                <div ref={messagesEndRef} />
              </>
            ) : (
              <div className="no-conversation">
                <div className="no-conversation-content">
                  <h3>Welcome to Sanskrit Engine</h3>
                  <p>
                    Start a conversation to explore Sanskrit text processing,
                    grammatical analysis, and reasoning capabilities.
                  </p>
                  <div className="feature-highlights">
                    <div className="feature">
                      <span className="feature-icon">📝</span>
                      <span>Sanskrit text processing</span>
                    </div>
                    <div className="feature">
                      <span className="feature-icon">🔍</span>
                      <span>Grammatical analysis</span>
                    </div>
                    <div className="feature">
                      <span className="feature-icon">🧠</span>
                      <span>Reasoning & inference</span>
                    </div>
                    <div className="feature">
                      <span className="feature-icon">🔗</span>
                      <span>Cross-domain mapping</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <MessageInput
            value={inputValue}
            onChange={setInputValue}
            onSend={handleSendMessage}
            onKeyPress={handleKeyPress}
            disabled={isLoading || !currentConversation}
            placeholder={
              currentConversation
                ? 'Type your message... (Enter to send, Shift+Enter for new line)'
                : 'Select or create a conversation to start chatting'
            }
          />
        </div>

        <div className="chat-sidebar">
          <ConversationList />
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;