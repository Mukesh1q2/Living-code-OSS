import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

// Mock the child components to avoid complex setup
jest.mock('./components/Header', () => {
  return function MockHeader() {
    return <header data-testid="header">Header</header>;
  };
});

jest.mock('./components/ChatInterface', () => {
  return function MockChatInterface() {
    return <div data-testid="chat-interface">Chat Interface</div>;
  };
});

jest.mock('./components/CodeEditor', () => {
  return function MockCodeEditor() {
    return <div data-testid="code-editor">Code Editor</div>;
  };
});

jest.mock('./components/DiagramCanvas', () => {
  return function MockDiagramCanvas() {
    return <div data-testid="diagram-canvas">Diagram Canvas</div>;
  };
});

// Mock WebSocket
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
};

(global as any).WebSocket = jest.fn(() => mockWebSocket);

describe('App', () => {
  test('renders without crashing', () => {
    render(<App />);
    expect(screen.getByTestId('header')).toBeInTheDocument();
  });

  test('renders chat interface by default', () => {
    render(<App />);
    expect(screen.getByTestId('chat-interface')).toBeInTheDocument();
  });

  test('has proper app structure', () => {
    const { container } = render(<App />);
    expect(container.querySelector('.App')).toBeInTheDocument();
  });
});