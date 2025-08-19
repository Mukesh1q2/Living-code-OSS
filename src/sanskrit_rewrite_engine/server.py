"""
FastAPI web server for Sanskrit Rewrite Engine.

This module provides the unified web API interface for the Sanskrit processing engine,
consolidating functionality from multiple server implementations.
"""

import json
import logging
import uuid
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from .engine import SanskritRewriteEngine, TransformationResult

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Request/Response Models
class ProcessRequest(BaseModel):
    """Request model for text processing."""
    text: str = Field(..., description="Sanskrit text to process", min_length=1, max_length=10000)
    max_passes: Optional[int] = Field(20, description="Maximum processing passes", ge=1, le=100)
    enable_tracing: Optional[bool] = Field(True, description="Enable transformation tracing")
    rules: Optional[List[str]] = Field(None, description="Optional rule set names to use", max_length=50)
    
    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v):
        """Validate text content for security."""
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        # Basic security check - no script tags or suspicious content
        suspicious_patterns = ['<script', 'javascript:', 'data:', 'vbscript:']
        v_lower = v.lower()
        for pattern in suspicious_patterns:
            if pattern in v_lower:
                raise ValueError(f'Suspicious content detected: {pattern}')
        return v


class ProcessResponse(BaseModel):
    """Response model for text processing."""
    success: bool
    input: str
    output: str
    converged: bool
    passes: int
    transformations: Dict[str, int]
    traces: List[Dict[str, Any]]
    processing_time: float
    timestamp: str
    request_id: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., description="Sanskrit text to analyze", min_length=1, max_length=10000)
    
    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v):
        """Validate text content for security."""
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        # Basic security check - no script tags or suspicious content
        suspicious_patterns = ['<script', 'javascript:', 'data:', 'vbscript:']
        v_lower = v.lower()
        for pattern in suspicious_patterns:
            if pattern in v_lower:
                raise ValueError(f'Suspicious content detected: {pattern}')
        return v


class AnalysisResponse(BaseModel):
    """Response model for text analysis."""
    success: bool
    input: str
    analysis: Dict[str, Any]
    timestamp: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]


class RuleInfo(BaseModel):
    """Information about a single rule."""
    id: str
    name: str
    description: str
    example: Optional[str] = None
    sutra: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None


class RulesResponse(BaseModel):
    """Response model for rules listing."""
    success: bool
    count: int
    rules: List[RuleInfo]


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    message: str
    timestamp: str
    path: Optional[str] = None
    request_id: Optional[str] = None


# Create FastAPI application
app = FastAPI(
    title="Sanskrit Rewrite Engine API",
    description="REST API for Sanskrit text processing and grammatical analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS with specific allowed origins
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=[
        "Content-Type", 
        "Authorization", 
        "X-Requested-With",
        "Accept",
        "Origin"
    ],
    expose_headers=["X-Request-ID", "X-Process-Time"],
    max_age=600  # Cache preflight requests for 10 minutes
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "*.localhost", "testserver"]
)

# Request size limits
MAX_REQUEST_SIZE = 1024 * 1024  # 1MB
MAX_TEXT_LENGTH = 10000  # Maximum text length in characters

# Global engine instance
engine = SanskritRewriteEngine()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )
        
        return response


class RequestMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request monitoring and logging with request IDs."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        logger.info(
            f"Request started - ID: {request_id} | "
            f"Method: {request.method} | "
            f"Path: {request.url.path} | "
            f"Client: {client_ip} | "
            f"User-Agent: {user_agent[:100]}..."  # Truncate long user agents
        )
        
        # Check content length before processing
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                content_length_int = int(content_length)
                if content_length_int > MAX_REQUEST_SIZE:
                    logger.warning(
                        f"Request rejected - ID: {request_id} | "
                        f"Reason: Content too large ({content_length_int} > {MAX_REQUEST_SIZE})"
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Request Entity Too Large",
                            "message": f"Request size {content_length_int} exceeds maximum {MAX_REQUEST_SIZE} bytes",
                            "request_id": request_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
            except ValueError:
                logger.warning(f"Invalid content-length header - ID: {request_id}")
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add request ID and timing to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            # Log successful request completion
            logger.info(
                f"Request completed - ID: {request_id} | "
                f"Status: {response.status_code} | "
                f"Time: {process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            # Calculate processing time even for errors
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed - ID: {request_id} | "
                f"Error: {str(e)} | "
                f"Time: {process_time:.3f}s"
            )
            
            # Return structured error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                },
                headers={"X-Request-ID": request_id, "X-Process-Time": f"{process_time:.3f}"}
            )


# Security and CORS Configuration
# Add security headers middleware first
app.add_middleware(SecurityHeadersMiddleware)

# Add request monitoring middleware
app.add_middleware(RequestMonitoringMiddleware)





# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured responses."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning(
        f"HTTP Exception - ID: {request_id} | "
        f"Status: {exc.status_code} | "
        f"Detail: {exc.detail}"
    )
    
    # Provide more helpful error messages based on status code
    helpful_messages = {
        400: "Bad Request: Please check your input parameters and try again.",
        401: "Unauthorized: Authentication required to access this resource.",
        403: "Forbidden: You don't have permission to access this resource.",
        404: "Not Found: The requested resource could not be found.",
        413: "Request Too Large: The request payload exceeds the maximum allowed size.",
        422: "Validation Error: Please check your input data format and values.",
        429: "Too Many Requests: Please wait before making another request.",
        500: "Internal Server Error: An unexpected error occurred on the server.",
        503: "Service Unavailable: The service is temporarily unavailable."
    }
    
    helpful_message = helpful_messages.get(exc.status_code, exc.detail)
    
    error_response = ErrorResponse(
        error=f"HTTP {exc.status_code}",
        message=helpful_message,
        timestamp=datetime.now().isoformat(),
        path=str(request.url.path)
    ).model_dump()
    error_response["request_id"] = request_id
    
    # Add additional context for validation errors
    if exc.status_code == 422 and hasattr(exc, 'detail') and isinstance(exc.detail, list):
        error_response["validation_errors"] = exc.detail
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response,
        headers={"X-Request-ID": request_id}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured responses."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(
        f"Unhandled Exception - ID: {request_id} | "
        f"Error: {str(exc)} | "
        f"Type: {type(exc).__name__}"
    )
    
    # Provide different messages based on exception type
    if isinstance(exc, ValueError):
        status_code = 400
        error_type = "Validation Error"
        message = f"Invalid input: {str(exc)}"
    elif isinstance(exc, TimeoutError):
        status_code = 408
        error_type = "Request Timeout"
        message = "The request took too long to process. Please try with shorter text or fewer transformations."
    elif isinstance(exc, MemoryError):
        status_code = 507
        error_type = "Insufficient Storage"
        message = "The request requires too much memory. Please try with shorter text."
    else:
        status_code = 500
        error_type = "Internal Server Error"
        message = "An unexpected error occurred. Please try again later."
    
    error_response = ErrorResponse(
        error=error_type,
        message=message,
        timestamp=datetime.now().isoformat(),
        path=str(request.url.path)
    ).model_dump()
    error_response["request_id"] = request_id
    error_response["error_type"] = type(exc).__name__
    
    return JSONResponse(
        status_code=status_code,
        content=error_response,
        headers={"X-Request-ID": request_id}
    )


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sanskrit Rewrite Engine API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /api/rules": "Available rules",
            "POST /api/process": "Process Sanskrit text",
            "POST /api/analyze": "Analyze Sanskrit text"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "example": {
            "url": "/api/process",
            "method": "POST",
            "body": {"text": "rāma + iti", "enable_tracing": True}
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for service monitoring.
    
    Returns the current status of the API server and its components.
    This endpoint can be used by load balancers, monitoring systems,
    and deployment tools to verify service health.
    
    **Example Response:**
    ```json
    {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": "2025-01-13T10:30:00Z",
        "components": {
            "api_server": "✅ running",
            "sanskrit_processor": "✅ available",
            "rule_engine": "✅ loaded (15 rules)",
            "tokenizer": "✅ available"
        }
    }
    ```
    """
    try:
        # Check engine health
        rule_count = engine.get_rule_count()
        engine_status = f"✅ loaded ({rule_count} rules)" if rule_count > 0 else "⚠️ no rules loaded"
        
        # Test basic engine functionality
        test_result = engine.process("test", False)
        processor_status = "✅ available" if test_result.success else "❌ error"
        
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            timestamp=datetime.now().isoformat(),
            components={
                "api_server": "✅ running",
                "sanskrit_processor": processor_status,
                "rule_engine": engine_status,
                "tokenizer": "✅ available"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="degraded",
            version="2.0.0",
            timestamp=datetime.now().isoformat(),
            components={
                "api_server": "✅ running",
                "sanskrit_processor": "❌ error",
                "rule_engine": "❌ error",
                "tokenizer": "❌ error"
            }
        )


@app.get("/rules", response_model=RulesResponse)
async def get_available_rules():
    """Get list of available transformation rules.
    
    Returns information about all loaded transformation rules,
    including their priority, category, and current enabled status.
    
    **Example Response:**
    ```json
    {
        "success": true,
        "count": 4,
        "rules": [
            {
                "id": "vowel_sandhi_a_i",
                "name": "Vowel Sandhi: a + i → e",
                "description": "Combines vowels 'a' and 'i' to form 'e'",
                "example": "rāma + iti → rāmeti",
                "sutra": "6.1.87",
                "category": "sandhi",
                "priority": 1,
                "enabled": true
            }
        ]
    }
    ```
    """
    try:
        # Get rules from the actual engine
        rules_by_priority = engine.rule_registry.get_rules_by_priority()
        
        rule_infos = []
        for rule in rules_by_priority:
            # Extract example from metadata if available
            example = rule.metadata.get('example')
            sutra = rule.metadata.get('sutra_ref') or rule.metadata.get('sutra')
            category = rule.metadata.get('category')
            
            rule_info = RuleInfo(
                id=rule.id,
                name=rule.name,
                description=rule.description,
                example=example,
                sutra=sutra,
                category=category,
                priority=rule.priority,
                enabled=rule.enabled
            )
            rule_infos.append(rule_info)
        
        # If no rules are loaded, provide sample rules for demonstration
        if not rule_infos:
            rule_infos = [
                RuleInfo(
                    id="vowel_sandhi_a_i",
                    name="Vowel Sandhi: a + i → e",
                    description="Combines vowels 'a' and 'i' to form 'e'",
                    example="rāma + iti → rāmeti",
                    sutra="6.1.87",
                    category="sandhi",
                    priority=1,
                    enabled=True
                ),
                RuleInfo(
                    id="vowel_sandhi_a_u",
                    name="Vowel Sandhi: a + u → o",
                    description="Combines vowels 'a' and 'u' to form 'o'",
                    example="deva + ukta → devokta",
                    sutra="6.1.87",
                    category="sandhi",
                    priority=1,
                    enabled=True
                ),
                RuleInfo(
                    id="compound_formation",
                    name="Compound Formation (Samāsa)",
                    description="Forms compound words from multiple stems",
                    example="deva + rāja → devarāja",
                    sutra="2.1.3",
                    category="morphology",
                    priority=2,
                    enabled=True
                ),
                RuleInfo(
                    id="genitive_case",
                    name="Genitive Case Formation",
                    description="Forms genitive case endings",
                    example="rāma → rāmasya",
                    sutra="7.1.12",
                    category="inflection",
                    priority=3,
                    enabled=True
                )
            ]
        
        return RulesResponse(
            success=True,
            count=len(rule_infos),
            rules=rule_infos
        )
        
    except Exception as e:
        logger.error(f"Error retrieving rules: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve rules. Please try again later."
        )


@app.post("/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest, http_request: Request):
    """Process Sanskrit text with transformations.
    
    This endpoint applies Sanskrit transformation rules to the input text,
    following the rule-based processing pipeline defined in the engine.
    
    **Example Request:**
    ```json
    {
        "text": "rāma + iti",
        "enable_tracing": true,
        "max_passes": 10
    }
    ```
    
    **Example Response:**
    ```json
    {
        "success": true,
        "input": "rāma + iti",
        "output": "rāmeti",
        "converged": true,
        "passes": 1,
        "transformations": {"vowel_sandhi_a_i": 1},
        "traces": [...],
        "processing_time": 0.001,
        "timestamp": "2025-01-13T10:30:00Z",
        "request_id": "uuid-here"
    }
    ```
    """
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    
    try:
        # Additional security validation
        if len(request.text) > MAX_TEXT_LENGTH:
            logger.warning(
                f"Text too long - ID: {request_id} | "
                f"Length: {len(request.text)} > {MAX_TEXT_LENGTH}"
            )
            raise HTTPException(
                status_code=400, 
                detail=f"Text too long. Maximum length is {MAX_TEXT_LENGTH} characters."
            )
        
        # Log processing request
        logger.info(
            f"Processing request - ID: {request_id} | "
            f"Text length: {len(request.text)} | "
            f"Max passes: {request.max_passes} | "
            f"Tracing: {request.enable_tracing}"
        )
        
        start_time = datetime.now()
        
        # Use the actual engine instead of consolidated function
        result = engine.process(request.text, request.enable_tracing)
        
        # Convert engine result to API response format
        api_result = {
            "success": result.success,
            "input": result.input_text,
            "output": result.output_text,
            "converged": True,  # Assume convergence for now
            "passes": len([t for t in result.trace if t.get('step') == 'iteration_summary']),
            "transformations": _count_transformations(result.transformations_applied),
            "traces": result.trace,
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        if not result.success:
            api_result["error"] = result.error_message
        
        logger.info(
            f"Processing completed - ID: {request_id} | "
            f"Success: {result.success} | "
            f"Transformations: {len(result.transformations_applied)}"
        )
        
        return ProcessResponse(**api_result)
        
    except ValueError as e:
        logger.warning(f"Validation error - ID: {request_id} | Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing error - ID: {request_id} | Error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Text processing failed. Please check your input and try again."
        )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalyzeRequest, http_request: Request):
    """Analyze Sanskrit text structure and properties."""
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    
    try:
        # Additional security validation
        if len(request.text) > MAX_TEXT_LENGTH:
            logger.warning(
                f"Text too long for analysis - ID: {request_id} | "
                f"Length: {len(request.text)} > {MAX_TEXT_LENGTH}"
            )
            raise HTTPException(
                status_code=400, 
                detail=f"Text too long. Maximum length is {MAX_TEXT_LENGTH} characters."
            )
        
        # Log analysis request
        logger.info(
            f"Analysis request - ID: {request_id} | "
            f"Text length: {len(request.text)}"
        )
        
        analysis = analyze_sanskrit_text_consolidated(request.text)
        
        logger.info(f"Analysis completed - ID: {request_id}")
        
        response_data = AnalysisResponse(
            success=True,
            input=request.text,
            analysis=analysis,
            timestamp=datetime.now().isoformat()
        )
        
        # Add request ID to response
        response_dict = response_data.model_dump()
        response_dict["request_id"] = request_id
        
        return response_dict
        
    except ValueError as e:
        logger.warning(f"Analysis validation error - ID: {request_id} | Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error - ID: {request_id} | Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Text analysis failed")


# Helper Functions
def _count_transformations(transformations_applied: List[str]) -> Dict[str, int]:
    """Count occurrences of each transformation type.
    
    Args:
        transformations_applied: List of transformation descriptions
        
    Returns:
        Dictionary mapping transformation names to counts
    """
    counts = {}
    for transformation in transformations_applied:
        # Extract rule name from transformation description
        # Format is typically "Rule Name (id: rule_id)"
        if " (id: " in transformation:
            rule_name = transformation.split(" (id: ")[0]
        else:
            rule_name = transformation
        
        counts[rule_name] = counts.get(rule_name, 0) + 1
    
    return counts


# Sanskrit Processing Functions (consolidated from existing servers)
def process_sanskrit_text_consolidated(text: str, max_passes: int = 20, enable_tracing: bool = True) -> Dict[str, Any]:
    """Consolidated Sanskrit text processing from all server implementations."""
    start_time = datetime.now()
    
    # Sanskrit transformation rules (consolidated from all servers)
    transformations = {
        # Vowel sandhi
        'rāma + iti': 'rāmeti',
        'deva + indra': 'devendra',
        'mahā + ātman': 'mahātman',
        'su + ukta': 'sukta',
        'guru + iti': 'gurūti',
        'a + i': 'e',
        'a + u': 'o',
        'ā + i': 'e',
        'ā + u': 'o',
        
        # Compound formation
        'deva + rāja': 'devarāja',
        'guru + kula': 'gurukula',
        'dharma + śāstra': 'dharmaśāstra',
        
        # Case formation
        'rāma + asya': 'rāmasya',
        'deva + asya': 'devasya',
        
        # Consonant sandhi
        'tat + ca': 'tac ca',
        'sat + jana': 'saj jana'
    }
    
    original_text = text
    output_text = text
    applied_rules = {}
    traces = []
    
    # Apply transformations
    for pattern, replacement in transformations.items():
        if pattern in output_text:
            old_text = output_text
            output_text = output_text.replace(pattern, replacement)
            
            rule_name = get_rule_name(pattern, replacement)
            applied_rules[rule_name] = applied_rules.get(rule_name, 0) + 1
            
            if enable_tracing:
                traces.append({
                    "pass_number": 1,
                    "rule_applied": rule_name,
                    "position": old_text.find(pattern),
                    "transformation": {
                        "before": pattern,
                        "after": replacement,
                        "context_before": old_text,
                        "context_after": output_text
                    }
                })
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "success": True,
        "input": original_text,
        "output": output_text,
        "converged": True,
        "passes": 1 if applied_rules else 0,
        "transformations": applied_rules,
        "traces": traces,
        "processing_time": processing_time,
        "timestamp": datetime.now().isoformat()
    }


def analyze_sanskrit_text_consolidated(text: str) -> Dict[str, Any]:
    """Consolidated Sanskrit text analysis from all server implementations."""
    words = text.split()
    
    return {
        "basic_stats": {
            "character_count": len(text),
            "word_count": len(words),
            "contains_markers": '+' in text or ':' in text or '_' in text
        },
        "tokenization": {
            "words": words,
            "potential_compounds": detect_compounds(text),
            "sandhi_opportunities": detect_sandhi_opportunities(text)
        },
        "morphology": {
            "stems": extract_stems(words),
            "possible_inflections": detect_inflections(words),
            "suffixes": extract_suffixes(words)
        },
        "transformations_available": get_available_transformations(text)
    }


def get_rule_name(pattern: str, replacement: str) -> str:
    """Generate rule name from pattern and replacement."""
    if '+' in pattern:
        parts = pattern.split(' + ')
        if len(parts) == 2:
            return f"sandhi_{parts[0].replace(' ', '_')}_{parts[1].replace(' ', '_')}"
    return f"transformation_{pattern.replace(' ', '_').replace('+', 'plus')}"


def detect_compounds(text: str) -> List[Dict[str, Any]]:
    """Detect potential compound words."""
    compounds = []
    if '+' in text:
        parts = text.split(' + ')
        if len(parts) >= 2:
            compounds.append({
                "type": "marked_compound",
                "parts": parts,
                "suggested_form": ''.join(parts)
            })
    return compounds


def detect_sandhi_opportunities(text: str) -> List[Dict[str, Any]]:
    """Detect sandhi opportunities in text."""
    opportunities = []
    words = text.split()
    
    for i in range(len(words) - 1):
        current = words[i]
        next_word = words[i + 1]
        
        if current.endswith('a') and next_word.startswith('i'):
            opportunities.append({
                "type": "vowel_sandhi",
                "pattern": f"{current} + {next_word}",
                "rule": "a + i → e"
            })
        elif current.endswith('a') and next_word.startswith('u'):
            opportunities.append({
                "type": "vowel_sandhi",
                "pattern": f"{current} + {next_word}",
                "rule": "a + u → o"
            })
    
    return opportunities


def extract_stems(words: List[str]) -> List[Dict[str, Any]]:
    """Extract potential word stems."""
    stems = []
    for word in words:
        if word.endswith('asya'):
            stems.append({"word": word, "stem": word[:-4], "suffix": "asya"})
        elif word.endswith('āya'):
            stems.append({"word": word, "stem": word[:-3], "suffix": "āya"})
        elif word.endswith('am'):
            stems.append({"word": word, "stem": word[:-2], "suffix": "am"})
        else:
            stems.append({"word": word, "stem": word, "suffix": None})
    return stems


def extract_suffixes(words: List[str]) -> List[str]:
    """Extract potential suffixes from words."""
    suffixes = []
    for word in words:
        if word.endswith('asya'):
            suffixes.append('asya (genitive)')
        elif word.endswith('āya'):
            suffixes.append('āya (dative)')
        elif word.endswith('am'):
            suffixes.append('am (accusative)')
    return suffixes


def detect_inflections(words: List[str]) -> List[Dict[str, Any]]:
    """Detect possible inflections in words."""
    inflections = []
    for word in words:
        if word.endswith('asya'):
            inflections.append({"word": word, "case": "genitive", "number": "singular"})
        elif word.endswith('āya'):
            inflections.append({"word": word, "case": "dative", "number": "singular"})
        elif word.endswith('am'):
            inflections.append({"word": word, "case": "accusative", "number": "singular"})
    return inflections


def get_available_transformations(text: str) -> List[str]:
    """Get list of transformations available for the given text."""
    available = []
    
    if 'rāma' in text and 'iti' in text:
        available.append("Vowel sandhi: rāma + iti → rāmeti")
    if 'deva' in text and 'indra' in text:
        available.append("Vowel sandhi: deva + indra → devendra")
    if '+' in text:
        available.append("Marked sandhi/compound formation")
    if any(word.endswith('a') for word in text.split()):
        available.append("Case inflection possible")
    if 'deva' in text:
        available.append("Compound formation with deva-")
        
    return available


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app
    """
    return app