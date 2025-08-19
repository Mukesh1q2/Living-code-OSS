# Sanskrit Rewrite Engine API Server

A comprehensive FastAPI backend server for the Sanskrit Rewrite Engine, providing REST endpoints, WebSocket support, and real-time streaming capabilities.

## Features

- **Sanskrit Text Processing**: Complete pipeline for Sanskrit text transformation
- **Chat Interface**: Conversational AI with Sanskrit reasoning capabilities
- **Rule Tracing**: Detailed debugging and analysis of rule applications
- **File Operations**: Secure file read/write operations with sandbox protection
- **Code Execution**: Safe code execution environment with resource limits
- **WebSocket Support**: Real-time communication and streaming responses
- **Authentication**: Bearer token authentication system
- **Comprehensive Logging**: Detailed audit trails and execution logs

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Core Dependencies

- `fastapi>=0.104.0` - Modern web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `pydantic>=2.0.0` - Data validation
- `python-multipart>=0.0.6` - File upload support
- `websockets>=11.0.0` - WebSocket support
- `psutil>=5.9.0` - System monitoring

## Quick Start

1. **Start the server:**
   ```bash
   python run_server.py
   ```

2. **Access the API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

3. **Test the endpoints:**
   ```bash
   python test_api.py
   ```

## API Endpoints

### Core Endpoints

#### Health Check
```
GET /health
```
Returns server health status and component availability.

#### Sanskrit Processing
```
POST /api/v1/process
```
Process Sanskrit text through the complete transformation pipeline.

**Request Body:**
```json
{
  "text": "namaste",
  "enable_tracing": true,
  "max_passes": 20,
  "options": {}
}
```

#### Chat Interface
```
POST /api/v1/chat
```
Interactive chat with Sanskrit reasoning capabilities.

**Request Body:**
```json
{
  "message": "Hello, can you help me with Sanskrit?",
  "conversation_id": "optional-uuid",
  "context": {},
  "stream": false
}
```

#### Rule Tracing
```
POST /api/v1/trace
```
Detailed tracing of rule applications for debugging.

**Request Body:**
```json
{
  "text": "sanskrit text",
  "rule_ids": ["optional", "rule", "filter"],
  "detail_level": "full"
}
```

### File Operations

#### File Management
```
POST /api/v1/files
```
Secure file operations with sandbox protection.

**Request Body:**
```json
{
  "operation": "read|write|delete|list",
  "file_path": "relative/path/to/file",
  "content": "file content for write operations"
}
```

### Code Execution

#### Safe Code Execution
```
POST /api/v1/execute
```
Execute code in a secure sandbox environment.

**Request Body:**
```json
{
  "code": "print('Hello World!')",
  "language": "python",
  "timeout": 30,
  "context": {}
}
```

### Real-time Features

#### WebSocket Connection
```
WS /ws/{client_id}
```
Real-time bidirectional communication.

#### Streaming Chat
```
GET /api/v1/stream/chat/{conversation_id}?message=your_message
```
Server-sent events for streaming chat responses.

#### File Upload
```
POST /api/v1/upload
```
Upload and process files with automatic Sanskrit text processing.

## Authentication

All API endpoints (except `/health`) require authentication using Bearer tokens:

```bash
curl -H "Authorization: Bearer your-token" http://localhost:8000/api/v1/process
```

## Configuration

The server supports various configuration options through environment variables:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (default: info)
- `RELOAD`: Enable auto-reload in development (default: true)

## Architecture

The API server is built with a modular architecture:

```
api_server.py           # Main FastAPI application
├── Pydantic Models     # Request/response validation
├── Authentication      # Bearer token auth
├── Component Manager   # Sanskrit engine components
├── Connection Manager  # WebSocket management
├── Error Handling      # Comprehensive error handling
└── Fallback Systems    # Graceful degradation
```

## Component Integration

The server integrates with multiple Sanskrit engine components:

- **Tokenizer**: Sanskrit text tokenization
- **Panini Engine**: Rule-based transformations
- **Semantic Pipeline**: Complete processing pipeline
- **Reasoning Core**: Logic programming backend
- **Hybrid Reasoner**: Multi-model reasoning
- **Safe Execution**: Secure code execution
- **MCP Server**: File operations with security

## Security Features

- **Sandbox Execution**: Isolated code execution environment
- **Resource Limits**: CPU, memory, and time constraints
- **File Access Control**: Restricted file system access
- **Input Validation**: Comprehensive request validation
- **Audit Logging**: Complete execution audit trails
- **Error Handling**: Secure error reporting

## Development

### Running in Development Mode

```bash
# With auto-reload
python run_server.py

# Or directly with uvicorn
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run API tests
python test_api.py

# Install test dependencies
pip install httpx pytest

# Run with pytest (if test files exist)
pytest tests/
```

### Logging

The server provides comprehensive logging:

- **Console Output**: Real-time server logs
- **File Logging**: Persistent logs in `api_server.log`
- **Audit Trails**: Execution logs in `execution_audit.log`

## Error Handling

The server implements graceful error handling:

- **Component Failures**: Fallback to basic functionality
- **Missing Dependencies**: Clear error messages
- **Resource Limits**: Proper timeout and limit enforcement
- **Security Violations**: Safe error reporting

## Performance

- **Async Operations**: Full async/await support
- **Connection Pooling**: Efficient WebSocket management
- **Resource Monitoring**: Real-time resource usage tracking
- **Caching**: Response caching where appropriate

## Deployment

### Production Deployment

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run_server.py"]
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies with `pip install -r requirements.txt`
2. **Port Conflicts**: Change port in `run_server.py` or use environment variable
3. **Permission Errors**: Ensure proper file system permissions
4. **Component Failures**: Check logs for specific component initialization errors

### Debug Mode

Enable debug logging by setting the log level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include proper logging
4. Update documentation
5. Add tests for new endpoints

## License

This API server is part of the Sanskrit Rewrite Engine project.