import React, { useRef, useEffect } from 'react';
import './MessageInput.css';

interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: (message: string) => void;
  onKeyPress?: (e: React.KeyboardEvent) => void;
  disabled?: boolean;
  placeholder?: string;
}

const MessageInput: React.FC<MessageInputProps> = ({
  value,
  onChange,
  onSend,
  onKeyPress,
  disabled = false,
  placeholder = 'Type your message...',
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  }, [value]);

  const handleSend = () => {
    if (value.trim() && !disabled) {
      onSend(value.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (onKeyPress) {
      onKeyPress(e);
    }
  };

  return (
    <div className="message-input">
      <div className="input-container">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className="message-textarea"
          rows={1}
          aria-label="Message input"
        />
        
        <div className="input-actions">
          <button
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            className="send-button"
            title="Send message (Enter)"
            aria-label="Send message"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22,2 15,22 11,13 2,9"></polygon>
            </svg>
          </button>
        </div>
      </div>
      
      <div className="input-footer">
        <div className="input-hints">
          <span className="hint">
            <kbd>Enter</kbd> to send, <kbd>Shift</kbd> + <kbd>Enter</kbd> for new line
          </span>
        </div>
        
        <div className="input-tools">
          <button
            className="tool-button"
            title="Insert Sanskrit text"
            disabled={disabled}
          >
            🕉️
          </button>
          <button
            className="tool-button"
            title="Code snippet"
            disabled={disabled}
          >
            &lt;/&gt;
          </button>
          <button
            className="tool-button"
            title="Mathematical expression"
            disabled={disabled}
          >
            ∑
          </button>
        </div>
      </div>
    </div>
  );
};

export default MessageInput;