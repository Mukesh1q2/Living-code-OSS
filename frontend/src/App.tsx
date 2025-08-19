import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Header from './components/Header';
import ChatInterface from './components/ChatInterface';
import CodeEditor from './components/CodeEditor';
import DiagramCanvas from './components/DiagramCanvas';
import IntegratedCanvas from './components/IntegratedCanvas';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { ChatProvider } from './contexts/ChatContext';

function App() {
  return (
    <WebSocketProvider>
      <ChatProvider>
        <Router>
          <div className="App">
            <Header />
            <main className="main-content">
              <Routes>
                <Route path="/" element={<ChatInterface />} />
                <Route path="/editor" element={<CodeEditor />} />
                <Route path="/canvas" element={<DiagramCanvas />} />
                <Route path="/integrated" element={<IntegratedCanvas />} />
              </Routes>
            </main>
          </div>
        </Router>
      </ChatProvider>
    </WebSocketProvider>
  );
}

export default App;