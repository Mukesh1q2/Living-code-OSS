#!/usr/bin/env python3
"""
Simple Backend Server for Sanskrit Rewrite Engine
A lightweight server that works without complex dependencies
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import FastAPI, fall back to simple HTTP server if not available
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available, falling back to simple HTTP server")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class ProcessRequest(BaseModel):
    text: str
    max_passes: Optional[int] = 20
    enable_tracing: Optional[bool] = True

class ProcessResponse(BaseModel):
    success: bool
    input: str
    output: str
    converged: bool
    passes: int
    transformations: Dict[str, int]
    traces: List[Dict[str, Any]]
    processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]

# Simple Sanskrit processing functions (mock implementations)
def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization - split by spaces and common separators"""
    import re
    # Basic tokenization
    tokens = re.findall(r'\S+', text)
    return tokens

def simple_sandhi_processing(text: str) -> str:
    """Simple sandhi rules - basic transformations"""
    # Basic sandhi rules
    transformations = {
        'rāma + iti': 'rāmeti',
        'deva + indra': 'devendra', 
        'mahā + ātman': 'mahātman',
        'su + ukta': 'sukta',
        'tat + ca': 'tac ca',
        'a + i': 'e',
        'a + u': 'o',
        'ā + i': 'e',
        'ā + u': 'o'
    }
    
    # Apply transformations
    result = text
    applied_rules = {}
    
    for pattern, replacement in transformations.items():
        if pattern in result:
            result = result.replace(pattern, replacement)
            rule_name = f"sandhi_{pattern.replace(' + ', '_')}"
            applied_rules[rule_name] = 1
    
    return result, applied_rules

def process_sanskrit_text_simple(text: str, max_passes: int = 20, enable_tracing: bool = True) -> Dict[str, Any]:
    """Simple Sanskrit text processing"""
    start_time = datetime.now()
    
    # Tokenize
    tokens = simple_tokenize(text)
    
    # Apply basic transformations
    output_text, applied_rules = simple_sandhi_processing(text)
    
    # Create traces if enabled
    traces = []
    if enable_tracing and applied_rules:
        traces.append({
            'pass_number': 1,
            'transformations': [
                {
                    'rule_name': rule_name,
                    'position': 0,
                    'before': text,
                    'after': output_text
                }
                for rule_name in applied_rules.keys()
            ]
        })
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return {
        'success': True,
        'input': text,
        'output': output_text,
        'converged': True,
        'passes': 1 if applied_rules else 0,
        'transformations': applied_rules,
        'traces': traces,
        'processing_time': processing_time,
        'timestamp': datetime.now().isoformat()
    }

if FASTAPI_AVAILABLE:
    # FastAPI implementation
    app = FastAPI(
        title="Sanskrit Rewrite Engine API",
        description="REST API for Sanskrit text processing and grammatical analysis",
        version="2.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"message": "Sanskrit Rewrite Engine API", "version": "2.0.0"}

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            timestamp=datetime.now().isoformat(),
            components={
                "tokenizer": "available",
                "rule_engine": "available", 
                "api_server": "running"
            }
        )

    @app.post("/api/process", response_model=ProcessResponse)
    async def process_text(request: ProcessRequest):
        try:
            result = process_sanskrit_text_simple(
                request.text,
                request.max_passes,
                request.enable_tracing
            )
            return ProcessResponse(**result)
        except Exception as e:
            logger.error(f"Processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/rules")
    async def get_available_rules():
        return {
            "rules": [
                {"name": "sandhi_a_i", "description": "a + i → e"},
                {"name": "sandhi_a_u", "description": "a + u → o"},
                {"name": "compound_formation", "description": "Form compounds"},
                {"name": "morphological_analysis", "description": "Analyze word structure"}
            ]
        }

    @app.post("/api/analyze")
    async def analyze_text(request: ProcessRequest):
        try:
            # Basic analysis
            tokens = simple_tokenize(request.text)
            result = process_sanskrit_text_simple(request.text)
            
            return {
                "success": True,
                "analysis": {
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "transformations": result['transformations'],
                    "morphological": {"stems": [], "suffixes": []},
                    "compounds": {"detected": [], "type": "unknown"},
                    "semantic": {"concepts": [], "relations": []}
                }
            }
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def run_fastapi_server():
        """Run FastAPI server"""
        logger.info("Starting FastAPI server on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

else:
    # Fallback HTTP server implementation
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    class SanskritAPIHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_json_response({"message": "Sanskrit Rewrite Engine API", "version": "2.0.0"})
            elif self.path == '/health':
                response = {
                    "status": "healthy",
                    "version": "2.0.0", 
                    "timestamp": datetime.now().isoformat(),
                    "components": {"api_server": "running"}
                }
                self.send_json_response(response)
            elif self.path == '/api/rules':
                rules = {
                    "rules": [
                        {"name": "sandhi_a_i", "description": "a + i → e"},
                        {"name": "sandhi_a_u", "description": "a + u → o"}
                    ]
                }
                self.send_json_response(rules)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == '/api/process':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                try:
                    request_data = json.loads(post_data.decode('utf-8'))
                    text = request_data.get('text', '')
                    
                    result = process_sanskrit_text_simple(text)
                    self.send_json_response(result)
                except Exception as e:
                    self.send_error(500, str(e))
            else:
                self.send_error(404)

        def send_json_response(self, data):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

    def run_simple_server():
        """Run simple HTTP server"""
        server = HTTPServer(('localhost', 8000), SanskritAPIHandler)
        logger.info("Starting simple HTTP server on http://localhost:8000")
        server.serve_forever()

def main():
    """Main entry point"""
    print("🚀 Starting Sanskrit Rewrite Engine Backend Server...")
    print("=" * 60)
    
    if FASTAPI_AVAILABLE:
        print("✅ FastAPI available - starting full-featured server")
        run_fastapi_server()
    else:
        print("⚠️  FastAPI not available - starting simple HTTP server")
        print("   Install FastAPI for full features: pip install fastapi uvicorn")
        run_simple_server()

if __name__ == "__main__":
    main()