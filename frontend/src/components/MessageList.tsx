import React from 'react';
import { ChatMessage } from '../contexts/ChatContext';
import MessageBubble from './MessageBubble';
import './MessageList.css';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading?: boolean;
}

const MessageList: React.FC<MessageListProps> = ({ messages, isLoading }) => {
  return (
    <div className="message-list">
      {messages.length === 0 ? (
        <div className="empty-messages">
          <div className="empty-messages-content">
            <span className="empty-icon">💬</span>
            <h4>Start the conversation</h4>
            <p>Ask about Sanskrit grammar, text processing, or reasoning capabilities.</p>
            <div className="example-prompts">
              <button className="example-prompt">
                "Analyze this Sanskrit text: राम गच्छति"
              </button>
              <button className="example-prompt">
                "Explain sandhi rules for vowel combinations"
              </button>
              <button className="example-prompt">
                "Generate code from Sanskrit description"
              </button>
            </div>
          </div>
        </div>
      ) : (
        <>
          {messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              isConsecutive={
                index > 0 && messages[index - 1].role === message.role
              }
            />
          ))}
          
          {isLoading && (
            <div className="typing-indicator">
              <div className="typing-bubble">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default MessageList;