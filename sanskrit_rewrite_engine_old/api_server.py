"""
FastAPI Backend Server for Sanskrit Rewrite Engine

This module provides a comprehensive REST API and WebSocket interface for the
Sanskrit Rewrite Engine, including chat functionality, rule tracing, file operations,
and real-time streaming responses.

Requirements: 13.1, 13.2, 13.3
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import Sanskrit engine components (with fallbacks for missing modules)
try:
    from .tokenizer import SanskritTokenizer
except ImportError:
    SanskritTokenizer = None

try:
    from .panini_engine import PaniniRuleEngine, PaniniEngineResult
except ImportError:
    PaniniRuleEngine = None
    PaniniEngineResult = None

try:
    from .essential_sutras import create_essential_sutras
except ImportError:
    def create_essential_sutras():
        return []

try:
    from .semantic_pipeline import process_sanskrit_text, ProcessingResult
except ImportError:
    async def process_sanskrit_text(*args, **kwargs):
        return type('ProcessingResult', (), {
            'output_text': kwargs.get('text', ''),
            'tokens': [],
            'transformations': [],
            'semantic_graph': {},
            'trace_data': {},
            'passes': 1
        })()

try:
    from .reasoning_core import ReasoningCore
except ImportError:
    ReasoningCore = None

try:
    from .hybrid_reasoning import HybridSanskritReasoner, ModelType, TaskComplexity
except ImportError:
    HybridSanskritReasoner = None
    ModelType = None
    TaskComplexity = None

from .safe_execution import CodeExecutionManager, ExecutionContext, create_safe_execution_manager

try:
    from .mcp_server import SanskritMCPServer, SecurityConfig, WorkspaceConfig
except ImportError:
    SanskritMCPServer = None
    SecurityConfig = None
    WorkspaceConfig = None

try:
    from .r_zero_integration import SanskritProblem, SanskritProblemType, SanskritDifficultyLevel
except ImportError:
    SanskritProblem = None
    SanskritProblemType = None
    SanskritDifficultyLevel = None

logger = logging.getLogger(__name__)

# Pydantic models for API
class SanskritProcessRequest(BaseModel):
    """Request model for Sanskrit text processing."""
    text: str = Field(..., description="Sanskrit text to process")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")
    enable_tracing: bool = Field(default=True, description="Enable detailed tracing")
    max_passes: int = Field(default=20, description="Maximum processing passes")

class SanskritProcessResponse(BaseModel):
    """Response model for Sanskrit text processing."""
    input_text: str
    output_text: str
    tokens: List[Dict[str, Any]]
    transformations: List[Dict[str, Any]]
    semantic_graph: Dict[str, Any]
    success: bool
    processing_time_ms: float
    trace_data: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    """Chat message model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    stream: bool = Field(default=False, description="Enable streaming response")

class ChatResponse(BaseModel):
    """Chat response model."""
    message: ChatMessage
    conversation_id: str
    response_time_ms: float
    model_used: str
    confidence: float

class RuleTraceRequest(BaseModel):
    """Rule trace request model."""
    text: str = Field(..., description="Text to trace")
    rule_ids: Optional[List[str]] = Field(None, description="Specific rule IDs to trace")
    detail_level: str = Field(default="full", description="Trace detail level: minimal, standard, full")

class RuleTraceResponse(BaseModel):
    """Rule trace response model."""
    input_text: str
    trace_data: Dict[str, Any]
    rule_applications: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]

class FileOperation(BaseModel):
    """File operation model."""
    operation: str = Field(..., description="Operation type: read, write, delete, list")
    file_path: str = Field(..., description="File path relative to workspace")
    content: Optional[str] = Field(None, description="File content for write operations")
    options: Dict[str, Any] = Field(default_factory=dict, description="Operation options")

class FileOperationResponse(BaseModel):
    """File operation response model."""
    success: bool
    operation: str
    file_path: str
    content: Optional[str] = None
    files: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class CodeExecutionRequest(BaseModel):
    """Code execution request model."""
    code: str = Field(..., description="Code to execute")
    language: str = Field(default="python", description="Programming language")
    timeout: int = Field(default=30, description="Execution timeout in seconds")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")

class CodeExecutionResponse(BaseModel):
    """Code execution response model."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: float
    language: str

# Connection manager for WebSocket connections
class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_connections: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Disconnect a WebSocket client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients."""
        for connection in self.active_connections.values():
            await connection.send_text(message)

# Initialize FastAPI app
app = FastAPI(
    title="Sanskrit Rewrite Engine API",
    description="Comprehensive API for Sanskrit text processing, reasoning, and code generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
connection_manager = ConnectionManager()
security = HTTPBearer()

# Global state
sanskrit_engine = None
reasoning_core = None
hybrid_reasoner = None
execution_manager = None
mcp_server = None
conversation_history: Dict[str, List[ChatMessage]] = {}

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication - extend as needed."""
    # For now, just return a basic user object
    return {"user_id": "default_user", "token": credentials.credentials}

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup."""
    global sanskrit_engine, reasoning_core, hybrid_reasoner, execution_manager, mcp_server
    
    logger.info("Starting Sanskrit Rewrite Engine API server...")
    
    try:
        # Initialize tokenizer and engine
        if SanskritTokenizer and PaniniRuleEngine:
            tokenizer = SanskritTokenizer()
            rule_engine = PaniniRuleEngine()
            rule_engine.load_rules(create_essential_sutras())
            sanskrit_engine = (tokenizer, rule_engine)
            logger.info("Sanskrit engine initialized")
        else:
            sanskrit_engine = None
            logger.warning("Sanskrit engine components not available")
        
        # Initialize reasoning core
        if ReasoningCore:
            reasoning_core = ReasoningCore()
            logger.info("Reasoning core initialized")
        else:
            reasoning_core = None
            logger.warning("Reasoning core not available")
        
        # Initialize hybrid reasoner
        if HybridSanskritReasoner:
            hybrid_reasoner = HybridSanskritReasoner()
            logger.info("Hybrid reasoner initialized")
        else:
            hybrid_reasoner = None
            logger.warning("Hybrid reasoner not available")
        
        # Initialize safe execution manager
        execution_manager = create_safe_execution_manager(str(Path.cwd()))
        logger.info("Safe execution manager initialized")
        
        # Initialize MCP server
        if SanskritMCPServer and SecurityConfig and WorkspaceConfig:
            security_config = SecurityConfig(
                allowed_directories=[Path.cwd()],
                max_file_size=10 * 1024 * 1024,  # 10MB
                allowed_extensions={".py", ".txt", ".md", ".json", ".yaml"}
            )
            workspace_config = WorkspaceConfig(
                workspace_root=Path.cwd(),
                temp_directory=Path.cwd() / "temp"
            )
            mcp_server = SanskritMCPServer(security_config, workspace_config)
            logger.info("MCP server initialized")
        else:
            mcp_server = None
            logger.warning("MCP server components not available")
        
        logger.info("API server initialization completed")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        # Don't raise - allow server to start with limited functionality
        logger.warning("Server starting with limited functionality")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Sanskrit Rewrite Engine API server...")
    
    # Close all WebSocket connections
    for client_id in list(connection_manager.active_connections.keys()):
        connection_manager.disconnect(client_id)
    
    # Cleanup execution manager
    if execution_manager:
        await execution_manager.cleanup()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "sanskrit_engine": sanskrit_engine is not None,
            "reasoning_core": reasoning_core is not None,
            "hybrid_reasoner": hybrid_reasoner is not None,
            "execution_manager": execution_manager is not None,
            "mcp_server": mcp_server is not None
        }
    }

# Sanskrit processing endpoints
@app.post("/api/v1/process", response_model=SanskritProcessResponse)
async def process_sanskrit_text_endpoint(
    request: SanskritProcessRequest,
    user = Depends(get_current_user)
):
    """Process Sanskrit text through the complete pipeline."""
    start_time = datetime.now()
    
    try:
        if not sanskrit_engine:
            raise HTTPException(status_code=503, detail="Sanskrit processing engine not available")
        
        tokenizer, rule_engine = sanskrit_engine
        
        # Process through semantic pipeline
        result = await process_sanskrit_text(
            text=request.text,
            tokenizer=tokenizer,
            rule_engine=rule_engine,
            enable_tracing=request.enable_tracing,
            max_passes=request.max_passes,
            options=request.options
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Handle token serialization safely
        tokens = []
        if hasattr(result, 'tokens') and result.tokens:
            for token in result.tokens:
                if hasattr(token, 'to_dict'):
                    tokens.append(token.to_dict())
                else:
                    tokens.append(str(token))
        
        return SanskritProcessResponse(
            input_text=request.text,
            output_text=getattr(result, 'output_text', request.text),
            tokens=tokens,
            transformations=getattr(result, 'transformations', []),
            semantic_graph=getattr(result, 'semantic_graph', {}),
            success=True,
            processing_time_ms=processing_time,
            trace_data=getattr(result, 'trace_data', {}) if request.enable_tracing else None
        )
        
    except Exception as e:
        logger.error(f"Error processing Sanskrit text: {e}")
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SanskritProcessResponse(
            input_text=request.text,
            output_text="",
            tokens=[],
            transformations=[],
            semantic_graph={},
            success=False,
            processing_time_ms=processing_time,
            trace_data={"error": str(e)}
        )

# Chat endpoints
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user = Depends(get_current_user)
):
    """Handle chat requests with Sanskrit reasoning."""
    start_time = datetime.now()
    
    try:
        # Get or create conversation
        conversation_id = request.conversation_id or str(uuid.uuid4())
        if conversation_id not in conversation_history:
            conversation_history[conversation_id] = []
        
        # Add user message to history
        user_message = ChatMessage(
            role="user",
            content=request.message,
            metadata=request.context
        )
        conversation_history[conversation_id].append(user_message)
        
        # Process with hybrid reasoner or fallback
        if hybrid_reasoner:
            response_content = await hybrid_reasoner.process_query(
                query=request.message,
                context=request.context,
                conversation_history=[msg.dict() for msg in conversation_history[conversation_id]]
            )
            
            response_text = response_content.response
            model_used = response_content.model_used
            confidence = response_content.confidence
        else:
            # Simple fallback response
            response_text = f"Echo: {request.message} (Sanskrit reasoning engine not available)"
            model_used = "fallback"
            confidence = 0.5
        
        # Create assistant response
        assistant_message = ChatMessage(
            role="assistant",
            content=response_text,
            metadata={
                "model_used": model_used,
                "confidence": confidence,
                "reasoning_steps": []
            }
        )
        conversation_history[conversation_id].append(assistant_message)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ChatResponse(
            message=assistant_message,
            conversation_id=conversation_id,
            response_time_ms=processing_time,
            model_used=model_used,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user = Depends(get_current_user)
):
    """Get conversation history."""
    if conversation_id not in conversation_history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": [msg.dict() for msg in conversation_history[conversation_id]]
    }

# Rule tracing endpoints
@app.post("/api/v1/trace", response_model=RuleTraceResponse)
async def trace_rules_endpoint(
    request: RuleTraceRequest,
    user = Depends(get_current_user)
):
    """Trace rule applications for debugging."""
    start_time = datetime.now()
    
    try:
        if not sanskrit_engine:
            raise HTTPException(status_code=503, detail="Sanskrit tracing engine not available")
        
        tokenizer, rule_engine = sanskrit_engine
        
        # Enable detailed tracing
        result = await process_sanskrit_text(
            text=request.text,
            tokenizer=tokenizer,
            rule_engine=rule_engine,
            enable_tracing=True,
            trace_detail_level=request.detail_level,
            rule_filter=request.rule_ids
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RuleTraceResponse(
            input_text=request.text,
            trace_data=getattr(result, 'trace_data', {}),
            rule_applications=getattr(result, 'transformations', []),
            performance_metrics={
                "processing_time_ms": processing_time,
                "passes": getattr(result, 'passes', 1),
                "rules_applied": len(getattr(result, 'transformations', []))
            }
        )
        
    except Exception as e:
        logger.error(f"Error tracing rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# File operation endpoints
@app.post("/api/v1/files", response_model=FileOperationResponse)
async def file_operations_endpoint(
    request: FileOperation,
    user = Depends(get_current_user)
):
    """Handle file operations with security sandbox."""
    try:
        if not mcp_server:
            # Fallback to basic file operations
            return await _basic_file_operations(request)
        
        if request.operation == "read":
            content = await mcp_server.read_file(request.file_path)
            return FileOperationResponse(
                success=True,
                operation="read",
                file_path=request.file_path,
                content=content
            )
        
        elif request.operation == "write":
            await mcp_server.write_file(request.file_path, request.content)
            return FileOperationResponse(
                success=True,
                operation="write",
                file_path=request.file_path
            )
        
        elif request.operation == "delete":
            await mcp_server.delete_file(request.file_path)
            return FileOperationResponse(
                success=True,
                operation="delete",
                file_path=request.file_path
            )
        
        elif request.operation == "list":
            files = await mcp_server.list_files(request.file_path)
            return FileOperationResponse(
                success=True,
                operation="list",
                file_path=request.file_path,
                files=files
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")
            
    except Exception as e:
        logger.error(f"Error in file operation: {e}")
        return FileOperationResponse(
            success=False,
            operation=request.operation,
            file_path=request.file_path,
            error=str(e)
        )

async def _basic_file_operations(request: FileOperation) -> FileOperationResponse:
    """Basic file operations fallback when MCP server is not available."""
    try:
        # Security check - only allow operations in current directory and subdirectories
        file_path = Path(request.file_path)
        if file_path.is_absolute() or ".." in str(file_path):
            raise ValueError("Invalid file path - only relative paths in workspace allowed")
        
        full_path = Path.cwd() / file_path
        
        if request.operation == "read":
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {request.file_path}")
            
            content = full_path.read_text(encoding='utf-8')
            return FileOperationResponse(
                success=True,
                operation="read",
                file_path=request.file_path,
                content=content
            )
        
        elif request.operation == "write":
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(request.content, encoding='utf-8')
            return FileOperationResponse(
                success=True,
                operation="write",
                file_path=request.file_path
            )
        
        elif request.operation == "delete":
            if full_path.exists():
                full_path.unlink()
            return FileOperationResponse(
                success=True,
                operation="delete",
                file_path=request.file_path
            )
        
        elif request.operation == "list":
            if not full_path.exists():
                raise FileNotFoundError(f"Directory not found: {request.file_path}")
            
            files = []
            for item in full_path.iterdir():
                files.append({
                    "name": item.name,
                    "path": str(item.relative_to(Path.cwd())),
                    "is_directory": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0
                })
            
            return FileOperationResponse(
                success=True,
                operation="list",
                file_path=request.file_path,
                files=files
            )
        
        else:
            raise ValueError(f"Unknown operation: {request.operation}")
            
    except Exception as e:
        return FileOperationResponse(
            success=False,
            operation=request.operation,
            file_path=request.file_path,
            error=str(e)
        )

# Code execution endpoints
@app.post("/api/v1/execute", response_model=CodeExecutionResponse)
async def execute_code_endpoint(
    request: CodeExecutionRequest,
    user = Depends(get_current_user)
):
    """Execute code in a safe sandbox environment."""
    start_time = datetime.now()
    
    try:
        if not execution_manager:
            raise HTTPException(status_code=503, detail="Code execution service not available")
        
        # Execute code using the manager
        result = execution_manager.execute_code(
            code=request.code,
            language=request.language,
            user_id=user.get("user_id"),
            timeout=request.timeout,
            **request.context
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return CodeExecutionResponse(
            success=result.success,
            output=result.output,
            error=result.error,
            execution_time_ms=execution_time,
            language=request.language
        )
        
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return CodeExecutionResponse(
            success=False,
            output="",
            error=str(e),
            execution_time_ms=execution_time,
            language=request.language
        )

# WebSocket endpoints
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication."""
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "chat":
                # Process chat message
                chat_request = ChatRequest(**message_data["data"])
                chat_response = await chat_endpoint(chat_request)
                
                # Send response back
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "chat_response",
                        "data": chat_response.dict()
                    }),
                    client_id
                )
            
            elif message_data.get("type") == "process":
                # Process Sanskrit text
                process_request = SanskritProcessRequest(**message_data["data"])
                process_response = await process_sanskrit_text_endpoint(process_request)
                
                # Send response back
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "process_response",
                        "data": process_response.dict()
                    }),
                    client_id
                )
            
            else:
                # Echo unknown messages
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "echo",
                        "data": message_data
                    }),
                    client_id
                )
                
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        connection_manager.disconnect(client_id)

# Streaming endpoints
@app.get("/api/v1/stream/chat/{conversation_id}")
async def stream_chat_response(
    conversation_id: str,
    message: str,
    user = Depends(get_current_user)
):
    """Stream chat responses for real-time interaction."""
    
    async def generate_response():
        try:
            # Get conversation history
            history = conversation_history.get(conversation_id, [])
            
            if hybrid_reasoner:
                # Stream response from hybrid reasoner
                async for chunk in hybrid_reasoner.stream_response(
                    query=message,
                    conversation_history=[msg.dict() for msg in history]
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
            else:
                # Fallback streaming response
                response_text = f"Echo: {message} (streaming not available)"
                for i, char in enumerate(response_text):
                    chunk = {
                        "type": "text",
                        "content": char,
                        "index": i,
                        "complete": i == len(response_text) - 1
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.05)  # Simulate streaming delay
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

# Upload endpoints
@app.post("/api/v1/upload")
async def upload_file(
    file: UploadFile = File(...),
    user = Depends(get_current_user)
):
    """Upload and process files."""
    try:
        # Read file content
        content = await file.read()
        
        # Save to secure location
        uploads_dir = Path.cwd() / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        file_path = f"uploads/{file.filename}"
        
        # Use MCP server if available, otherwise basic file operations
        if mcp_server:
            await mcp_server.write_file(file_path, content.decode('utf-8'))
        else:
            full_path = Path.cwd() / file_path
            full_path.write_bytes(content)
        
        # Process if it's a Sanskrit text file
        if file.filename.endswith(('.txt', '.md')) and sanskrit_engine:
            try:
                process_request = SanskritProcessRequest(text=content.decode('utf-8'))
                result = await process_sanskrit_text_endpoint(process_request, user)
                
                return {
                    "filename": file.filename,
                    "file_path": file_path,
                    "size": len(content),
                    "processed": True,
                    "result": result.dict()
                }
            except Exception as e:
                logger.warning(f"Failed to process uploaded file: {e}")
        
        return {
            "filename": file.filename,
            "file_path": file_path,
            "size": len(content),
            "processed": False
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )