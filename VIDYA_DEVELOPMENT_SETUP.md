# Vidya Quantum Interface - Development Setup

This document provides setup instructions for the Vidya quantum Sanskrit AI consciousness interface local development environment.

## Project Structure

```
vidya-quantum-interface/
├── vidya-frontend/          # React + TypeScript + Vite frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── services/        # WebSocket and API services
│   │   ├── types/           # Shared TypeScript interfaces
│   │   └── main.tsx         # Application entry point
│   ├── package.json
│   └── vite.config.ts
├── vidya-backend/           # FastAPI Python backend
│   ├── main.py              # FastAPI server with WebSocket support
│   └── requirements.txt     # Python dependencies
└── dev-server.py            # Concurrent development server launcher
```

## Prerequisites

- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **Git** for version control

## Quick Start

### Option 1: Concurrent Development Server (Recommended)

Run both frontend and backend servers simultaneously:

```bash
# From the project root directory
python dev-server.py
```

This will:
- Install Python backend dependencies automatically
- Install Node.js frontend dependencies automatically
- Start the FastAPI backend on http://localhost:8000
- Start the React frontend on http://localhost:3000
- Enable hot module replacement for both servers
- Provide unified logging output

### Option 2: Manual Setup

If you prefer to run servers separately:

#### Backend Setup

```bash
cd vidya-backend

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI development server
python main.py
```

Backend will be available at:
- API: http://localhost:8000
- WebSocket: ws://localhost:8000/ws
- API Documentation: http://localhost:8000/docs

#### Frontend Setup

```bash
cd vidya-frontend

# Install Node.js dependencies
npm install

# Start the Vite development server
npm run dev
```

Frontend will be available at:
- Application: http://localhost:3000
- Hot Module Replacement enabled

## Development Features

### Hot Module Replacement

- **Frontend**: Vite provides instant hot module replacement for React components
- **Backend**: FastAPI runs with `--reload` flag for automatic server restart on code changes
- **Shared Types**: TypeScript interfaces are shared between frontend and backend

### Real-time Communication

- **WebSocket Connection**: Automatic connection between frontend and backend
- **Message Broadcasting**: Real-time updates for consciousness state changes
- **Connection Recovery**: Automatic reconnection with message queuing

### Development Tools

- **React DevTools**: Browser extension support
- **FastAPI Docs**: Interactive API documentation at `/docs`
- **Performance Monitoring**: Built-in FPS and memory usage tracking
- **Debug Logging**: Comprehensive logging for both frontend and backend

## API Endpoints

### REST API

- `GET /` - Health check and server status
- `GET /api/consciousness` - Get current Vidya consciousness state
- `GET /api/neural-network` - Get neural network nodes
- `POST /api/consciousness/update` - Update consciousness state
- `POST /api/sanskrit/analyze` - Analyze Sanskrit text
- `GET /api/dev/status` - Development server status

### WebSocket Events

- `consciousness_update` - Vidya consciousness state changes
- `quantum_state_change` - Quantum state transitions
- `sanskrit_analysis` - Sanskrit text analysis results
- `neural_network_update` - Neural network modifications
- `user_input` - User interaction events
- `system_status` - Server status updates

## Shared TypeScript Interfaces

The `vidya-frontend/src/types/shared.ts` file contains all shared interfaces:

- **Quantum State Management**: `QuantumState`, `EntanglementInfo`
- **Consciousness Modeling**: `VidyaConsciousness`, `PersonalityProfile`
- **Neural Networks**: `NetworkNode`, `NetworkConnection`
- **Sanskrit Processing**: `SanskritToken`, `PaniniRule`
- **Communication**: `WebSocketMessage`, `ApiResponse`

## Three.js Integration

The frontend uses React Three Fiber for 3D visualizations:

- **Vidya Consciousness**: Animated Om symbol with quantum effects
- **Neural Network**: 3D network visualization with Sanskrit rule nodes
- **Quantum Effects**: Particle systems and shader-based visualizations
- **Performance Optimization**: Automatic quality adjustment based on device capabilities

## Development Workflow

1. **Start Development Server**: Run `python dev-server.py`
2. **Open Browser**: Navigate to http://localhost:3000
3. **Make Changes**: Edit frontend or backend code
4. **See Updates**: Changes are automatically reflected (hot reload)
5. **Test WebSocket**: Use browser dev tools to monitor WebSocket messages
6. **API Testing**: Use http://localhost:8000/docs for interactive API testing

## Troubleshooting

### Port Conflicts

If ports 3000 or 8000 are in use:

- **Frontend**: Edit `vidya-frontend/vite.config.ts` to change the port
- **Backend**: Edit `vidya-backend/main.py` to change the uvicorn port
- **WebSocket**: Update the WebSocket URL in `vidya-frontend/src/services/WebSocketManager.ts`

### Dependency Issues

```bash
# Backend dependencies
cd vidya-backend
pip install -r requirements.txt

# Frontend dependencies
cd vidya-frontend
rm -rf node_modules package-lock.json
npm install
```

### WebSocket Connection Issues

- Ensure backend server is running on port 8000
- Check browser console for WebSocket connection errors
- Verify CORS settings in `vidya-backend/main.py`

## Next Steps

After completing this setup, you can:

1. **Implement Sanskrit Engine Integration**: Connect to the existing Sanskrit rewrite engine
2. **Add LLM Services**: Integrate Hugging Face models for AI responses
3. **Enhance Quantum Effects**: Implement advanced WebGL shaders
4. **Expand Neural Network**: Add more sophisticated network visualizations
5. **Improve Consciousness**: Develop more complex personality and learning systems

## Performance Considerations

- **Development Mode**: Optimized for development with extensive logging and debugging
- **Production Ready**: Architecture designed for easy cloud deployment
- **Resource Management**: Automatic cleanup of Three.js objects and WebGL resources
- **Scalability**: WebSocket connection management supports multiple concurrent users

This setup provides a solid foundation for developing the Vidya quantum Sanskrit AI consciousness interface with hot reload, real-time communication, and comprehensive development tools.