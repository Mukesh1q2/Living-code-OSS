import React, { useState } from 'react';
import { ChatMessage } from '../contexts/ChatContext';
import './MessageBubble.css';

interface MessageBubbleProps {
  message: ChatMessage;
  isConsecutive?: boolean;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, isConsecutive }) => {
  const [showMetadata, setShowMetadata] = useState(false);

  const formatTime = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  // Detect Sanskrit text (basic heuristic)
  const containsSanskrit = (text: string) => {
    // Check for Devanagari characters
    return /[\u0900-\u097F]/.test(text);
  };

  const renderContent = (content: string) => {
    // Split content by code blocks
    const parts = content.split(/(```[\s\S]*?```|`[^`]+`)/);
    
    return parts.map((part, index) => {
      if (part.startsWith('```')) {
        // Code block
        const code = part.slice(3, -3);
        const lines = code.split('\n');
        const language = lines[0].trim();
        const codeContent = lines.slice(1).join('\n');
        
        return (
          <div key={index} className="code-block">
            {language && <div className="code-language">{language}</div>}
            <pre><code>{codeContent}</code></pre>
          </div>
        );
      } else if (part.startsWith('`') && part.endsWith('`')) {
        // Inline code
        return (
          <code key={index} className="inline-code">
            {part.slice(1, -1)}
          </code>
        );
      } else {
        // Regular text
        const className = containsSanskrit(part) ? 'sanskrit-text' : '';
        return (
          <span key={index} className={className}>
            {part}
          </span>
        );
      }
    });
  };

  if (isSystem) {
    return (
      <div className="message-bubble system-message">
        <div className="message-content">
          <div className="system-icon">ℹ️</div>
          <div className="system-text">{renderContent(message.content)}</div>
        </div>
        <div className="message-time">{formatTime(message.timestamp)}</div>
      </div>
    );
  }

  return (
    <div className={`message-bubble ${isUser ? 'user' : 'assistant'} ${isConsecutive ? 'consecutive' : ''}`}>
      <div className="message-wrapper">
        {!isConsecutive && (
          <div className="message-avatar">
            {isUser ? '👤' : '🤖'}
          </div>
        )}
        
        <div className="message-body">
          <div className="message-content">
            {renderContent(message.content)}
          </div>
          
          <div className="message-footer">
            <span className="message-time">
              {formatTime(message.timestamp)}
            </span>
            
            {message.metadata && Object.keys(message.metadata).length > 0 && (
              <button
                className="metadata-toggle"
                onClick={() => setShowMetadata(!showMetadata)}
                title="Show metadata"
              >
                ⓘ
              </button>
            )}
          </div>
          
          {showMetadata && message.metadata && (
            <div className="message-metadata">
              <div className="metadata-header">Message Details</div>
              <div className="metadata-content">
                {Object.entries(message.metadata).map(([key, value]) => (
                  <div key={key} className="metadata-item">
                    <span className="metadata-key">{key}:</span>
                    <span className="metadata-value">
                      {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;