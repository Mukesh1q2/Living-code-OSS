#!/usr/bin/env node

/**
 * Manual WebSocket Test Script
 * 
 * This script tests the WebSocket connection to the Vidya backend server.
 * Run with: node scripts/test-websocket.js
 */

const WebSocket = require('ws');

const WS_URL = 'ws://localhost:8000/ws';

console.log('🔌 Testing WebSocket connection to Vidya backend...');
console.log(`📡 Connecting to: ${WS_URL}`);

const ws = new WebSocket(WS_URL);

ws.on('open', () => {
  console.log('✅ Connected to Vidya WebSocket server!');
  
  // Send a ping message
  const pingMessage = {
    type: 'ping',
    payload: {},
    timestamp: new Date().toISOString(),
    id: `ping_${Date.now()}`
  };
  
  console.log('📤 Sending ping message...');
  ws.send(JSON.stringify(pingMessage));
  
  // Send a user input message
  setTimeout(() => {
    const userMessage = {
      type: 'user_input',
      payload: { input: 'Hello Vidya! How are you?' },
      timestamp: new Date().toISOString(),
      id: `user_${Date.now()}`
    };
    
    console.log('📤 Sending user input message...');
    ws.send(JSON.stringify(userMessage));
  }, 1000);
  
  // Send a Sanskrit analysis request
  setTimeout(() => {
    const sanskritMessage = {
      type: 'analyze_sanskrit',
      payload: { text: 'नमस्ते' },
      timestamp: new Date().toISOString(),
      id: `sanskrit_${Date.now()}`
    };
    
    console.log('📤 Sending Sanskrit analysis request...');
    ws.send(JSON.stringify(sanskritMessage));
  }, 2000);
  
  // Send a quantum state change
  setTimeout(() => {
    const quantumMessage = {
      type: 'quantum_state_change',
      payload: { change: 'superposition' },
      timestamp: new Date().toISOString(),
      id: `quantum_${Date.now()}`
    };
    
    console.log('📤 Sending quantum state change...');
    ws.send(JSON.stringify(quantumMessage));
  }, 3000);
});

ws.on('message', (data) => {
  try {
    const message = JSON.parse(data.toString());
    console.log(`📥 Received message [${message.type}]:`, {
      type: message.type,
      timestamp: message.timestamp,
      payload: typeof message.payload === 'object' 
        ? JSON.stringify(message.payload).substring(0, 100) + '...'
        : message.payload
    });
    
    // Handle specific message types
    switch (message.type) {
      case 'connection_established':
        console.log('🎉 Connection established! Vidya is ready.');
        break;
      case 'pong':
        console.log('🏓 Pong received - connection is alive!');
        break;
      case 'vidya_response':
        console.log('🤖 Vidya says:', message.payload.text);
        break;
      case 'sanskrit_analysis':
        console.log('📚 Sanskrit analysis completed:', {
          success: message.payload.success,
          tokens: message.payload.tokens?.length || 0,
          nodes: message.payload.network_nodes?.length || 0
        });
        break;
      case 'quantum_state_change':
        console.log('⚛️  Quantum state changed:', message.payload.change);
        break;
    }
  } catch (error) {
    console.error('❌ Error parsing message:', error);
    console.log('Raw message:', data.toString());
  }
});

ws.on('error', (error) => {
  console.error('❌ WebSocket error:', error.message);
  console.log('💡 Make sure the Vidya backend server is running on localhost:8000');
});

ws.on('close', (code, reason) => {
  console.log(`🔌 Connection closed [${code}]: ${reason || 'No reason provided'}`);
  process.exit(0);
});

// Handle process termination
process.on('SIGINT', () => {
  console.log('\n👋 Closing WebSocket connection...');
  ws.close();
});

// Auto-close after 30 seconds
setTimeout(() => {
  console.log('\n⏰ Test completed after 30 seconds');
  ws.close();
}, 30000);