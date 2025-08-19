# Sanskrit Rewrite Engine Frontend

A modern React-based frontend for the Sanskrit Rewrite Engine, providing an intuitive interface for Sanskrit text processing, grammatical analysis, and reasoning capabilities.

## Features

### 🗨️ ChatGPT-like Interface
- Real-time conversation with Sanskrit AI
- Conversation history and management
- Streaming responses for immediate feedback
- Sanskrit text rendering with proper fonts

### 📝 Code Editor
- Monaco Editor with Sanskrit syntax highlighting
- Multi-language support (Sanskrit, Python, JavaScript, etc.)
- Real-time error detection
- Code execution and output display
- Customizable themes and font sizes

### 🎨 Visualization Canvas
- Interactive diagrams for Sanskrit analysis
- Token visualization and relationships
- Rule application flow charts
- Semantic graph representations
- Zoom, pan, and export capabilities

### ♿ Accessibility Features
- WCAG 2.1 AA compliant
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support
- Proper ARIA labels and semantic HTML

## Technology Stack

- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Type-safe development
- **Monaco Editor** - VS Code editor component
- **D3.js** - Data visualization and interactive diagrams
- **Socket.IO** - Real-time WebSocket communication
- **Axios** - HTTP client for API communication
- **React Router** - Client-side routing
- **Jest & Testing Library** - Comprehensive testing suite

## Getting Started

### Prerequisites

- **Node.js 18+** and npm (or yarn/pnpm)
- **Sanskrit Rewrite Engine backend** running on port 8000
- **Git** for version control

### Quick Start

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server (with hot reload)
npm start

# Open browser to http://localhost:3000
```

### Development Workflow

#### 1. Initial Setup
```bash
# Clone repository and navigate to frontend
cd frontend

# Install dependencies
npm install

# Copy environment template
cp .env.example .env.local

# Start development server
npm start
```

#### 2. Development Commands
```bash
# Start development server with hot reload
npm start

# Run tests in watch mode
npm test

# Run tests with coverage
npm test -- --coverage

# Run linting
npm run lint

# Fix linting issues
npm run lint:fix

# Type checking
npm run type-check

# Build for production
npm run build

# Preview production build locally
npm run preview
```

#### 3. Environment Configuration

Create a `.env.local` file in the frontend directory:

```env
# Backend API Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000

# Development Settings
REACT_APP_DEBUG=true
REACT_APP_LOG_LEVEL=debug

# Feature Flags
REACT_APP_ENABLE_WEBSOCKETS=true
REACT_APP_ENABLE_MONACO_EDITOR=true
```

#### 4. Backend Integration

The frontend expects the Sanskrit Rewrite Engine backend to be running:

```bash
# In the root directory, start the backend
python -m sanskrit_rewrite_engine.server

# Or using the CLI
sanskrit-cli serve --host localhost --port 8000
```

#### 5. Proxy Configuration

The frontend is configured to proxy API requests to the backend:
- Development: `http://localhost:8000` (via package.json proxy)
- Production: Configure via `REACT_APP_API_URL` environment variable

## Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/
│   │   ├── ChatInterface.tsx      # Main chat interface
│   │   ├── CodeEditor.tsx         # Monaco-based code editor
│   │   ├── DiagramCanvas.tsx      # D3 visualization canvas
│   │   ├── Header.tsx             # Navigation header
│   │   ├── MessageBubble.tsx      # Individual chat messages
│   │   ├── MessageInput.tsx       # Message input component
│   │   ├── MessageList.tsx        # Message history display
│   │   └── ConversationList.tsx   # Conversation sidebar
│   ├── contexts/
│   │   ├── ChatContext.tsx        # Chat state management
│   │   └── WebSocketContext.tsx   # WebSocket connection
│   ├── services/
│   │   └── apiService.ts          # API client
│   ├── __tests__/                 # Test files
│   ├── App.tsx                    # Main application
│   └── index.tsx                  # Entry point
├── package.json
└── tsconfig.json
```

## Key Components

### ChatInterface
The main chat interface providing a ChatGPT-like experience for interacting with the Sanskrit AI. Features include:
- Conversation management
- Real-time messaging
- Sanskrit text rendering
- Message history
- Streaming responses

### CodeEditor
A powerful code editor built on Monaco Editor with:
- Sanskrit syntax highlighting
- Multi-language support
- Real-time execution
- Error detection
- Customizable appearance

### DiagramCanvas
Interactive visualization component using D3.js for:
- Token relationship diagrams
- Rule application flows
- Semantic graph visualization
- Interactive exploration
- Export capabilities

## API Integration

The frontend communicates with the backend through:

### REST API
- `/api/v1/chat` - Chat messages
- `/api/v1/process` - Sanskrit text processing
- `/api/v1/trace` - Rule tracing
- `/api/v1/execute` - Code execution
- `/api/v1/files` - File operations

### WebSocket
- Real-time chat streaming
- Live processing updates
- Collaborative features

## Testing

Comprehensive test suite covering:

```bash
# Run all tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run specific test file
npm test ChatInterface.test.tsx
```

### Test Categories
- **Unit Tests** - Individual component functionality
- **Integration Tests** - Component interactions
- **Accessibility Tests** - WCAG compliance
- **API Tests** - Service layer testing

## Accessibility

The frontend is built with accessibility as a priority:

- **Keyboard Navigation** - Full keyboard support
- **Screen Readers** - ARIA labels and semantic HTML
- **Color Contrast** - WCAG AA compliant colors
- **Focus Management** - Proper focus indicators
- **Alternative Text** - Descriptive labels for visual elements

## Performance

Optimizations include:

- **Code Splitting** - Lazy loading of routes
- **Memoization** - React.memo and useMemo
- **Virtual Scrolling** - Efficient large list rendering
- **Bundle Optimization** - Tree shaking and minification
- **Caching** - API response caching

## Build and Deployment

### Production Build

```bash
# Build for production
npm run build

# Preview production build locally
npm run preview

# Analyze bundle size
npm run build:analyze
```

### Automated Build Scripts

Use the provided build scripts for consistent builds:

```bash
# Unix/Linux/macOS
./scripts/build-frontend.sh

# Windows
scripts\build-frontend.bat
```

These scripts will:
- Check Node.js version requirements
- Install/update dependencies
- Run type checking and linting
- Execute tests with coverage
- Create optimized production build
- Provide build size information

### Deployment

```bash
# Deploy to production server (Unix/Linux)
./scripts/deploy-frontend.sh

# Or manually copy build files
cp -r build/* /var/www/sanskrit-rewrite-engine/
```

### Environment-Specific Builds

```bash
# Development build
npm run build

# Production build with optimizations
NODE_ENV=production npm run build

# Build with custom API endpoint
REACT_APP_API_URL=https://api.example.com npm run build
```

## Development Workflow

### Daily Development

1. **Start development environment:**
   ```bash
   # Quick setup (first time)
   ./scripts/dev-setup.sh  # Unix/Linux/macOS
   scripts\dev-setup.bat   # Windows
   
   # Daily workflow
   npm start
   ```

2. **Code quality checks:**
   ```bash
   npm run lint          # Check code style
   npm run type-check    # TypeScript validation
   npm test              # Run tests
   ```

3. **Before committing:**
   ```bash
   npm run lint:fix      # Fix linting issues
   npm run test:coverage # Ensure test coverage
   ```

### Troubleshooting

#### Common Issues

**Build fails with memory errors:**
```bash
# Increase Node.js memory limit
NODE_OPTIONS="--max-old-space-size=4096" npm run build
```

**Proxy not working:**
- Ensure backend is running on port 8000
- Check `REACT_APP_API_URL` in `.env.local`
- Verify `proxy` setting in `package.json`

**Dependencies out of sync:**
```bash
# Clean install
rm -rf node_modules package-lock.json
npm install
```

**TypeScript errors:**
```bash
# Check configuration
npm run type-check

# Clear TypeScript cache
rm -rf node_modules/.cache
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

### Development Guidelines

1. **Code Style:**
   - Follow existing TypeScript/React patterns
   - Use functional components with hooks
   - Implement proper error boundaries
   - Follow accessibility best practices

2. **Testing:**
   - Write unit tests for components
   - Include integration tests for user flows
   - Maintain test coverage above 80%
   - Test accessibility compliance

3. **Performance:**
   - Use React.memo for expensive components
   - Implement proper code splitting
   - Optimize bundle size
   - Monitor Core Web Vitals

4. **Documentation:**
   - Update README for new features
   - Document component props with TypeScript
   - Include usage examples
   - Update API documentation

### Pull Request Process

1. Create feature branch from `main`
2. Run full test suite: `npm run test:coverage`
3. Check code quality: `npm run lint && npm run type-check`
4. Build successfully: `npm run build`
5. Update documentation as needed
6. Submit pull request with clear description

## License

This project is part of the Sanskrit Rewrite Engine system.