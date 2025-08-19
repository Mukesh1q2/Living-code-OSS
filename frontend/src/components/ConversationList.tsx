import React from 'react';
import { useChat } from '../contexts/ChatContext';
import './ConversationList.css';

const ConversationList: React.FC = () => {
  const {
    conversations,
    currentConversation,
    createConversation,
    selectConversation,
    deleteConversation,
  } = useChat();

  const handleCreateConversation = () => {
    createConversation();
  };

  const handleDeleteConversation = (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this conversation?')) {
      deleteConversation(conversationId);
    }
  };

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (days === 1) {
      return 'Yesterday';
    } else if (days < 7) {
      return `${days} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const getLastMessage = (conversation: any) => {
    if (conversation.messages.length === 0) {
      return 'No messages yet';
    }
    
    const lastMessage = conversation.messages[conversation.messages.length - 1];
    const content = lastMessage.content;
    
    if (content.length > 50) {
      return content.substring(0, 50) + '...';
    }
    
    return content;
  };

  return (
    <div className="conversation-list">
      <div className="conversation-list-header">
        <h3>Conversations</h3>
        <button
          onClick={handleCreateConversation}
          className="btn btn-primary new-conversation-btn"
          title="New Conversation"
        >
          <span>+</span>
        </button>
      </div>

      <div className="conversation-items">
        {conversations.length === 0 ? (
          <div className="no-conversations">
            <p>No conversations yet</p>
            <button onClick={handleCreateConversation} className="btn btn-outline">
              Start your first conversation
            </button>
          </div>
        ) : (
          conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`conversation-item ${
                currentConversation?.id === conversation.id ? 'active' : ''
              }`}
              onClick={() => selectConversation(conversation.id)}
            >
              <div className="conversation-content">
                <div className="conversation-header">
                  <h4 className="conversation-title">{conversation.title}</h4>
                  <span className="conversation-date">
                    {formatDate(conversation.updatedAt)}
                  </span>
                </div>
                <p className="conversation-preview">
                  {getLastMessage(conversation)}
                </p>
                <div className="conversation-meta">
                  <span className="message-count">
                    {conversation.messages.length} messages
                  </span>
                </div>
              </div>
              <button
                onClick={(e) => handleDeleteConversation(e, conversation.id)}
                className="delete-conversation-btn"
                title="Delete Conversation"
              >
                <span>×</span>
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ConversationList;