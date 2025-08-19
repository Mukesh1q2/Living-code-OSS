# WebSocket Communication System Implementation

This document describes the WebSocket communication system implemented for the Vidya Quantum Interface, fulfilling task 5 from the implementation plan.

## Overview

The WebSocket system provides real-time, bidirectional communication between the React frontend and the FastAPI backend server. It includes robust error handling, automatic reconnection, message queuing, and development-friendly debugging tools.

## Architecture

### Core Components

1. **WebSocket Store** (`lib/websocket.ts`)
   - Zustand-based state management for WebSocket connection
   - Message queuing and history
   - Connection state management
   - Error recovery and reconnection logic

2. **WebSocket Provider** (`components/WebSocketProvider.tsx`)
   - React context provider for WebSocket functionality
   - Message handler registration
   - Integration with quantum state management
   - Connection status indicators

3. **Quantum WebSocket Integration** (`lib/useQuantumWebSocket.ts`)
   - Hooks for integrating WebSocket with quantum visualizations
   - Real-time quantum state synchronization
   - Sanskrit analysis visualization updates

4. **Debug Tools** (`components/WebSocketDebugPanel.tsx`)
   - Development debugging interface
   - Connection monitoring and testing
   - Message history and statistics

## Features Implemented

### ✅ Real-time Server Communication
- WebSocket client with automatic connection management
- Bidirectional message passing between frontend and backend
- Support for multiple concurrent connections

### ✅ Message Queuing and Error Recovery
- Automatic message queuing when disconnected
- Message replay after reconnection
- Exponential backoff for reconnection attempts
- Graceful error handling and recovery

### ✅ Connection State Management and Automatic Reconnection
- Connection state tracking (DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING, ERROR)
- Automatic reconnection on unexpected disconnections
- Configurable reconnection attempts and delays
- Manual connection control

### ✅ Typed Message Interfaces
- TypeScript interfaces for all message types
- Type-safe message handling
- Structured message format with timestamps and IDs

### ✅ Development-friendly Logging and Debugging
- Comprehensive debug logging in development mode
- WebSocket debug panel with connection monitoring
- Message history and statistics tracking
- Test utilities and manual testing tools

## Message Types

The system supports the following typed message interfaces:

```typescript
// Core message types
- ConsciousnessUpdateMessage
- SanskritAnalysisMessage  
- ProcessingUpdateMessage
- NeuralNetworkUpdateMessage
- QuantumStateChangeMessage
- VidyaResponseMessage
- ErrorMessage
- PingMessage / PongMessage
```

## Usage Examples

### Basic Connection
```typescript
import { useWebSocket } from '@/components/WebSocketProvider';

function MyComponent() {
  const { isConnected, sendMessage } = useWebSocket();
  
  const handleSendMessage = () => {
    sendMessage({
      type: 'user_input',
      payload: { input: 'Hello Vidya!' }
    });
  };
  
  return (
    <div>
      Status: {isConnected ? 'Connected' : 'Disconnected'}
      <button onClick={handleSendMessage}>Send Message</button>
    </div>
  );
}
```

### Quantum Integration
```typescript
import { useQuantumVisualization } from '@/lib/useQuantumWebSocket';

function QuantumComponent() {
  const { quantumState, triggerSuperposition } = useQuantumVisualization();
  
  return (
    <div>
      <button onClick={triggerSuperposition}>
        Trigger Superposition
      </button>
      {quantumState.superpositionActive && <div>In Superposition!</div>}
    </div>
  );
}
```

### Message Handling
```typescript
import { useWebSocketMessages } from '@/lib/websocket';

function MessageHandler() {
  const { addHandler, removeHandler } = useWebSocketMessages();
  
  useEffect(() => {
    const handler = (message) => {
      console.log('Received:', message);
    };
    
    addHandler('sanskrit_analysis', handler);
    return () => removeHandler('sanskrit_analysis');
  }, []);
}
```

## Configuration

### Environment Variables
```bash
# WebSocket URL (defaults to ws://localhost:8000/ws in development)
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws

# Enable debug mode
NODE_ENV=development
```

### WebSocket Provider Configuration
```typescript
<WebSocketProvider 
  autoConnect={true}
  debugMode={process.env.NODE_ENV === 'development'}
>
  {children}
</WebSocketProvider>
```

## Testing

### Unit Tests
```bash
npm run test:run -- lib/__tests__/websocket.test.ts
```

### Integration Tests (requires backend running)
```bash
npm run test:run -- lib/__tests__/integration.test.ts
```

### Manual Testing
```bash
node scripts/test-websocket.js
```

## Development Tools

### Debug Panel
- Access via the debug button (🔧) in the bottom-left corner (development only)
- View connection status, message history, and statistics
- Send test messages and monitor responses
- Connection controls and error information

### Debug Console
```typescript
// Available in browser console during development
WebSocketDebugger.getConnectionInfo()
WebSocketDebugger.sendTestMessage('ping')
WebSocketDebugger.getMessageStats()
```

### Connection Status Indicator
- Shows real-time connection status in the top-right corner (development only)
- Color-coded status: Green (connected), Yellow (connecting), Red (error)

## Backend Integration

The WebSocket client is designed to work with the FastAPI backend server running on `localhost:8000`. The backend provides the following endpoints:

- `GET /` - API information and available endpoints
- `GET /health` - Health check
- `POST /api/process` - Sanskrit text processing
- `WebSocket /ws` - Real-time communication

## Error Handling

The system includes comprehensive error handling:

1. **Connection Errors**: Automatic retry with exponential backoff
2. **Message Errors**: Graceful error reporting and recovery
3. **Network Issues**: Automatic reconnection and message queuing
4. **Server Errors**: Error message propagation to UI components

## Performance Considerations

- Message history limited to last 100 messages
- Automatic cleanup of disconnected WebSocket connections
- Efficient message queuing with retry limits
- Debounced reconnection attempts to prevent spam

## Security

- CORS configuration for local development
- Input validation for all message types
- Secure WebSocket connections (WSS) support for production
- Rate limiting considerations for message sending

## Future Enhancements

The WebSocket system is designed to support future enhancements:

1. **Authentication**: Token-based WebSocket authentication
2. **Rooms/Channels**: Multi-user support with message routing
3. **Compression**: Message compression for large payloads
4. **Metrics**: Advanced performance and usage metrics
5. **Clustering**: Support for multiple backend instances

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Ensure backend server is running on localhost:8000
   - Check CORS configuration
   - Verify WebSocket URL in configuration

2. **Messages Not Received**
   - Check message handler registration
   - Verify message type spelling
   - Enable debug mode for detailed logging

3. **Reconnection Issues**
   - Check network connectivity
   - Verify reconnection settings
   - Monitor debug panel for error details

### Debug Commands
```javascript
// In browser console
WebSocketDebugger.setDebugMode(true)
WebSocketDebugger.getConnectionInfo()
WebSocketDebugger.sendTestMessage('ping')
```

## Requirements Fulfilled

This implementation fulfills all requirements from task 5:

- ✅ **Create WebSocket client in React for real-time server communication**
- ✅ **Implement message queuing and error recovery for connection issues**  
- ✅ **Add connection state management and automatic reconnection**
- ✅ **Create typed message interfaces for different communication types**
- ✅ **Implement development-friendly logging and debugging for WebSocket messages**

The system provides a robust foundation for real-time communication between the Vidya quantum interface and the Sanskrit processing backend, enabling seamless integration of consciousness updates, Sanskrit analysis, and quantum visualization effects.