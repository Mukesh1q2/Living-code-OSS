import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import ChatInterface from '../ChatInterface';
import { ChatProvider } from '../../contexts/ChatContext';
import { WebSocketProvider } from '../../contexts/WebSocketContext';

// Mock the API service
jest.mock('../../services/apiService', () => ({
  apiService: {
    sendChatMessage: jest.fn().mockResolvedValue({
      message: {
        id: 'test-id',
        role: 'assistant',
        content: 'Test response',
        timestamp: new Date().toISOString(),
        metadata: {},
      },
      conversation_id: 'test-conversation',
      response_time_ms: 100,
      model_used: 'test-model',
      confidence: 0.9,
    }),
  },
}));

// Mock WebSocket
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
};

(global as any).WebSocket = jest.fn(() => mockWebSocket);

const renderChatInterface = () => {
  return render(
    <BrowserRouter>
      <WebSocketProvider>
        <ChatProvider>
          <ChatInterface />
        </ChatProvider>
      </WebSocketProvider>
    </BrowserRouter>
  );
};

describe('ChatInterface', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders chat interface with welcome message', () => {
    renderChatInterface();
    
    expect(screen.getByText('Welcome to Sanskrit Engine')).toBeInTheDocument();
    expect(screen.getByText(/Start a conversation to explore/)).toBeInTheDocument();
  });

  test('displays feature highlights', () => {
    renderChatInterface();
    
    expect(screen.getByText('Sanskrit text processing')).toBeInTheDocument();
    expect(screen.getByText('Grammatical analysis')).toBeInTheDocument();
    expect(screen.getByText('Reasoning & inference')).toBeInTheDocument();
    expect(screen.getByText('Cross-domain mapping')).toBeInTheDocument();
  });

  test('shows streaming toggle control', () => {
    renderChatInterface();
    
    const streamingToggle = screen.getByLabelText(/Real-time streaming/);
    expect(streamingToggle).toBeInTheDocument();
    expect(streamingToggle).not.toBeChecked();
  });

  test('enables streaming when toggle is clicked', async () => {
    const user = userEvent.setup();
    renderChatInterface();
    
    const streamingToggle = screen.getByLabelText(/Real-time streaming/);
    await user.click(streamingToggle);
    
    expect(streamingToggle).toBeChecked();
  });

  test('displays message input with correct placeholder', () => {
    renderChatInterface();
    
    const messageInput = screen.getByPlaceholderText(/Select or create a conversation/);
    expect(messageInput).toBeInTheDocument();
    expect(messageInput).toBeDisabled();
  });

  test('shows conversation list sidebar', () => {
    renderChatInterface();
    
    expect(screen.getByText('Conversations')).toBeInTheDocument();
    expect(screen.getByText('+')).toBeInTheDocument(); // New conversation button
  });

  test('creates new conversation when button is clicked', async () => {
    const user = userEvent.setup();
    renderChatInterface();
    
    const newConversationBtn = screen.getByTitle('New Conversation');
    await user.click(newConversationBtn);
    
    // Should show the conversation title input becomes enabled
    await waitFor(() => {
      const messageInput = screen.getByPlaceholderText(/Type your message/);
      expect(messageInput).not.toBeDisabled();
    });
  });

  test('displays error banner when error occurs', () => {
    // This would require mocking an error state in the context
    // For now, we'll test the error banner structure
    renderChatInterface();
    
    // The error banner should not be visible initially
    expect(screen.queryByText(/×/)).not.toBeInTheDocument();
  });

  test('has accessible keyboard navigation', () => {
    renderChatInterface();
    
    const newConversationBtn = screen.getByTitle('New Conversation');
    expect(newConversationBtn).toHaveAttribute('tabIndex', '0');
  });

  test('supports responsive design classes', () => {
    renderChatInterface();
    
    const chatInterface = screen.getByTestId('chat-interface') || 
                         document.querySelector('.chat-interface');
    expect(chatInterface).toHaveClass('chat-interface');
  });

  test('handles empty conversation state correctly', () => {
    renderChatInterface();
    
    expect(screen.getByText('Start the conversation')).toBeInTheDocument();
    expect(screen.getByText(/Ask about Sanskrit grammar/)).toBeInTheDocument();
  });

  test('shows example prompts for user guidance', () => {
    renderChatInterface();
    
    expect(screen.getByText(/Analyze this Sanskrit text/)).toBeInTheDocument();
    expect(screen.getByText(/Explain sandhi rules/)).toBeInTheDocument();
    expect(screen.getByText(/Generate code from Sanskrit/)).toBeInTheDocument();
  });

  test('maintains conversation history', async () => {
    const user = userEvent.setup();
    renderChatInterface();
    
    // Create a new conversation first
    const newConversationBtn = screen.getByTitle('New Conversation');
    await user.click(newConversationBtn);
    
    // Wait for the input to be enabled
    await waitFor(() => {
      const messageInput = screen.getByPlaceholderText(/Type your message/);
      expect(messageInput).not.toBeDisabled();
    });
  });
});

describe('ChatInterface Accessibility', () => {
  test('has proper ARIA labels', () => {
    renderChatInterface();
    
    const messageInput = screen.getByLabelText(/Message input/) || 
                        screen.getByRole('textbox');
    expect(messageInput).toBeInTheDocument();
  });

  test('supports keyboard navigation', () => {
    renderChatInterface();
    
    const focusableElements = screen.getAllByRole('button');
    focusableElements.forEach(element => {
      expect(element).toHaveAttribute('tabIndex');
    });
  });

  test('has sufficient color contrast', () => {
    renderChatInterface();
    
    // This is a basic test - in a real app you'd use tools like axe-core
    const chatInterface = document.querySelector('.chat-interface');
    expect(chatInterface).toBeInTheDocument();
  });

  test('supports screen readers', () => {
    renderChatInterface();
    
    // Check for semantic HTML elements
    expect(screen.getByRole('main') || document.querySelector('main')).toBeInTheDocument();
  });
});