# Frontend Development Guide

This guide covers the development workflow for the Sanskrit Rewrite Engine frontend application.

## Quick Start

### Prerequisites

- **Node.js 18+** and npm
- **Python 3.9+** (for backend)
- **Git** for version control

### Setup Commands

```bash
# Complete development setup
./scripts/dev-setup.sh      # Unix/Linux/macOS
scripts\dev-setup.bat       # Windows

# Manual setup
cd frontend
npm install
cp .env.example .env.local
npm start
```

## Project Structure

```
frontend/
├── public/                 # Static assets
│   ├── index.html         # HTML template
│   └── manifest.json      # PWA manifest
├── src/
│   ├── components/        # React components
│   │   ├── ChatInterface.tsx
│   │   ├── CodeEditor.tsx
│   │   ├── DiagramCanvas.tsx
│   │   └── ...
│   ├── contexts/          # React contexts
│   │   ├── ChatContext.tsx
│   │   └── WebSocketContext.tsx
│   ├── services/          # API and external services
│   │   └── apiService.ts
│   ├── __tests__/         # Test files
│   ├── App.tsx           # Main application component
│   └── index.tsx         # Application entry point
├── .env.example          # Environment template
├── .env.local           # Local environment (gitignored)
├── package.json         # Dependencies and scripts
├── tsconfig.json        # TypeScript configuration
└── README.md           # Frontend-specific documentation
```

## Development Workflow

### 1. Environment Setup

The frontend requires specific environment variables to connect to the backend:

```bash
# Copy and customize environment file
cp frontend/.env.example frontend/.env.local
```

Key environment variables:
- `REACT_APP_API_URL`: Backend API endpoint (default: http://localhost:8000)
- `REACT_APP_WS_URL`: WebSocket endpoint (default: ws://localhost:8000)
- `REACT_APP_DEBUG`: Enable debug logging (default: true)

### 2. Development Server

```bash
cd frontend

# Start development server with hot reload
npm start

# The application will open at http://localhost:3000
# API requests are proxied to http://localhost:8000
```

### 3. Code Quality

```bash
# Type checking
npm run type-check

# Linting
npm run lint
npm run lint:fix

# Testing
npm test                    # Interactive test runner
npm run test:coverage      # Run tests with coverage report
```

### 4. Building

```bash
# Development build
npm run build

# Production build with optimizations
NODE_ENV=production npm run build

# Automated build with quality checks
./scripts/build-frontend.sh    # Unix/Linux/macOS
scripts\build-frontend.bat     # Windows
```

## Backend Integration

### API Communication

The frontend communicates with the Sanskrit Rewrite Engine backend through:

1. **REST API** (via axios):
   - `/api/v1/process` - Text processing
   - `/api/v1/chat` - Chat interactions
   - `/api/v1/trace` - Rule tracing
   - `/api/v1/health` - Health checks

2. **WebSocket** (via socket.io-client):
   - Real-time chat streaming
   - Live processing updates
   - Collaborative features

### Proxy Configuration

Development proxy is configured in `package.json`:

```json
{
  "proxy": "http://localhost:8000"
}
```

This automatically proxies API requests from `http://localhost:3000/api/*` to `http://localhost:8000/api/*`.

### Starting the Backend

```bash
# From project root
source venv/bin/activate     # Unix/Linux/macOS
venv\Scripts\activate.bat    # Windows

# Start the backend server
sanskrit-cli serve --host localhost --port 8000

# Or directly with Python
python -m sanskrit_rewrite_engine.server
```

## Component Architecture

### Key Components

1. **ChatInterface**: Main chat UI with message history
2. **CodeEditor**: Monaco-based code editor with Sanskrit support
3. **DiagramCanvas**: D3.js visualizations for Sanskrit analysis
4. **MessageBubble**: Individual chat message rendering
5. **MessageInput**: Text input with Sanskrit keyboard support

### State Management

- **React Context**: For global state (chat, websocket)
- **Local State**: Component-specific state with hooks
- **Custom Hooks**: Reusable stateful logic

### Styling

- **CSS Modules**: Component-scoped styles
- **Tailwind CSS**: Utility-first CSS framework
- **Responsive Design**: Mobile-first approach
- **Accessibility**: WCAG 2.1 AA compliance

## Testing Strategy

### Test Types

1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Component interactions
3. **Accessibility Tests**: WCAG compliance
4. **API Tests**: Service layer testing

### Testing Tools

- **Jest**: Test runner and assertion library
- **React Testing Library**: Component testing utilities
- **jsdom**: DOM simulation for tests
- **Coverage**: Code coverage reporting

### Running Tests

```bash
# Interactive test runner
npm test

# Single test run with coverage
npm run test:coverage

# Specific test file
npm test ChatInterface.test.tsx

# Watch mode for specific tests
npm test -- --watch --testNamePattern="ChatInterface"
```

## Build and Deployment

### Build Process

The build process includes:

1. **Type Checking**: Validates TypeScript types
2. **Linting**: Code style and quality checks
3. **Testing**: Full test suite execution
4. **Bundling**: Webpack optimization and minification
5. **Asset Processing**: Image optimization, CSS extraction

### Build Scripts

```bash
# Automated build with all checks
./scripts/build-frontend.sh    # Unix/Linux/macOS
scripts\build-frontend.bat     # Windows

# Manual build steps
npm run type-check
npm run lint
npm run test:coverage
npm run build
```

### Deployment

```bash
# Automated deployment
./scripts/deploy-frontend.sh

# Manual deployment
cp -r frontend/build/* /var/www/sanskrit-rewrite-engine/
```

### Environment-Specific Builds

```bash
# Development
npm run build

# Staging
REACT_APP_API_URL=https://staging-api.example.com npm run build

# Production
REACT_APP_API_URL=https://api.example.com npm run build
```

## Performance Optimization

### Bundle Analysis

```bash
# Analyze bundle size
npm run build:analyze

# This opens webpack-bundle-analyzer in your browser
```

### Optimization Techniques

1. **Code Splitting**: Lazy loading of routes and components
2. **Memoization**: React.memo and useMemo for expensive operations
3. **Virtual Scrolling**: Efficient rendering of large lists
4. **Image Optimization**: WebP format and lazy loading
5. **Caching**: Service worker and API response caching

## Troubleshooting

### Common Issues

#### Build Failures

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json .eslintcache
npm install

# Increase memory for large builds
NODE_OPTIONS="--max-old-space-size=4096" npm run build
```

#### Proxy Issues

1. Ensure backend is running on port 8000
2. Check `REACT_APP_API_URL` in `.env.local`
3. Verify proxy configuration in `package.json`
4. Clear browser cache and restart dev server

#### TypeScript Errors

```bash
# Check TypeScript configuration
npm run type-check

# Clear TypeScript cache
rm -rf node_modules/.cache
npm install
```

#### Test Failures

```bash
# Update snapshots
npm test -- --updateSnapshot

# Clear Jest cache
npm test -- --clearCache

# Run tests in band (no parallel execution)
npm test -- --runInBand
```

### Debug Mode

Enable debug logging by setting environment variables:

```bash
# In .env.local
REACT_APP_DEBUG=true
REACT_APP_LOG_LEVEL=debug
```

## Contributing

### Code Style Guidelines

1. **TypeScript**: Use strict type checking
2. **Components**: Functional components with hooks
3. **Props**: Define interfaces for all component props
4. **Error Handling**: Implement error boundaries
5. **Accessibility**: Include ARIA labels and semantic HTML

### Git Workflow

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and commit: `git commit -m "Add new feature"`
3. Run quality checks: `npm run lint && npm run type-check && npm test`
4. Push and create pull request

### Code Review Checklist

- [ ] TypeScript types are properly defined
- [ ] Components are tested with good coverage
- [ ] Accessibility requirements are met
- [ ] Performance impact is considered
- [ ] Documentation is updated
- [ ] Build passes without warnings
- [ ] No console errors in browser

## Resources

### Documentation

- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [Monaco Editor](https://microsoft.github.io/monaco-editor/)
- [D3.js Documentation](https://d3js.org/)

### Tools

- [React Developer Tools](https://chrome.google.com/webstore/detail/react-developer-tools/)
- [Redux DevTools](https://chrome.google.com/webstore/detail/redux-devtools/)
- [Accessibility Insights](https://accessibilityinsights.io/)
- [Lighthouse](https://developers.google.com/web/tools/lighthouse)

This guide should help you get started with frontend development for the Sanskrit Rewrite Engine. For backend-specific information, refer to the main project documentation.