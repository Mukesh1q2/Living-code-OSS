#!/usr/bin/env python3
"""
Simple server starter for Sanskrit Rewrite Engine
Avoids import conflicts and complex dependencies
"""

import sys
import os
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("Starting Sanskrit Rewrite Engine Server...")
print("=" * 50)

class SanskritAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/':
                response = {
                    "message": "Sanskrit Rewrite Engine API",
                    "version": "2.0.0",
                    "status": "running",
                    "endpoints": [
                        "/health",
                        "/api/process",
                        "/api/rules",
                        "/api/analyze"
                    ]
                }
                self.send_json_response(response)
                
            elif self.path == '/health':
                response = {
                    "status": "healthy",
                    "version": "2.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "components": {
                        "api_server": "running",
                        "tokenizer": "available",
                        "rule_engine": "available"
                    }
                }
                self.send_json_response(response)
                
            elif self.path == '/api/rules':
                response = {
                    "success": True,
                    "rules": [
                        {
                            "id": "sandhi_a_i",
                            "name": "Vowel Sandhi a+i→e",
                            "description": "Combines 'a' and 'i' to form 'e'",
                            "example": "rāma + iti → rāmeti",
                            "sutra": "6.1.87"
                        },
                        {
                            "id": "sandhi_a_u", 
                            "name": "Vowel Sandhi a+u→o",
                            "description": "Combines 'a' and 'u' to form 'o'",
                            "example": "deva + ukta → devokta",
                            "sutra": "6.1.87"
                        },
                        {
                            "id": "compound_formation",
                            "name": "Compound Formation",
                            "description": "Forms compound words (samāsa)",
                            "example": "deva + rāja → devarāja",
                            "sutra": "2.1.3"
                        },
                        {
                            "id": "morphological_gen",
                            "name": "Genitive Case",
                            "description": "Forms genitive case endings",
                            "example": "rāma → rāmasya",
                            "sutra": "7.1.12"
                        }
                    ]
                }
                self.send_json_response(response)
                
            elif self.path == '/favicon.ico':
                # Handle favicon requests gracefully
                self.send_response(204)  # No Content
                self.end_headers()
                
            else:
                # Handle unknown endpoints gracefully
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                error_response = {"error": "Endpoint not found", "path": self.path}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
                
        except Exception as e:
            print(f"Error in GET request: {e}")
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                error_response = {"error": "Internal server error"}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
            except:
                pass  # Connection already closed
    
    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers.get('Content-Length', 0))
        
        if content_length > 0:
            post_data = self.rfile.read(content_length)
            try:
                request_data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return
        else:
            request_data = {}
        
        if self.path == '/api/process':
            self.handle_process_request(request_data)
        elif self.path == '/api/analyze':
            self.handle_analyze_request(request_data)
        else:
            self.send_error(404, "Endpoint not found")
    
    def handle_process_request(self, request_data):
        """Handle text processing requests"""
        text = request_data.get('text', '')
        max_passes = request_data.get('max_passes', 20)
        enable_tracing = request_data.get('enable_tracing', True)
        
        if not text:
            self.send_error(400, "No text provided")
            return
        
        try:
            # Simple Sanskrit processing
            result = self.process_sanskrit_simple(text, max_passes, enable_tracing)
            self.send_json_response(result)
        except Exception as e:
            self.send_error(500, f"Processing error: {str(e)}")
    
    def handle_analyze_request(self, request_data):
        """Handle text analysis requests"""
        text = request_data.get('text', '')
        
        if not text:
            self.send_error(400, "No text provided")
            return
        
        try:
            # Simple analysis
            tokens = text.split()
            result = {
                "success": True,
                "analysis": {
                    "input": text,
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "character_count": len(text),
                    "word_count": len([t for t in tokens if t.isalpha()]),
                    "morphological": {
                        "stems": self.extract_stems(tokens),
                        "suffixes": self.extract_suffixes(tokens)
                    },
                    "compounds": {
                        "detected": self.detect_compounds(text),
                        "type": "tatpuruṣa" if "+" in text else "unknown"
                    },
                    "transformations_available": self.get_available_transformations(text)
                }
            }
            self.send_json_response(result)
        except Exception as e:
            self.send_error(500, f"Analysis error: {str(e)}")
    
    def process_sanskrit_simple(self, text, max_passes=20, enable_tracing=True):
        """Simple Sanskrit text processing with basic rules"""
        start_time = datetime.now()
        
        # Basic sandhi transformations
        transformations = {
            'rāma + iti': 'rāmeti',
            'deva + indra': 'devendra',
            'mahā + ātman': 'mahātman',
            'su + ukta': 'sukta',
            'tat + ca': 'tac ca',
            'rāma + asya': 'rāmasya',
            'deva + rāja': 'devarāja',
            'guru + kula': 'gurukula'
        }
        
        # Apply transformations
        output_text = text
        applied_rules = {}
        traces = []
        
        for pattern, replacement in transformations.items():
            if pattern in output_text:
                old_text = output_text
                output_text = output_text.replace(pattern, replacement)
                rule_name = f"sandhi_{pattern.replace(' + ', '_').replace(' ', '_')}"
                applied_rules[rule_name] = 1
                
                if enable_tracing:
                    traces.append({
                        "pass_number": 1,
                        "transformations": [{
                            "rule_name": rule_name,
                            "rule_id": rule_name,
                            "position": old_text.find(pattern),
                            "before": old_text,
                            "after": output_text,
                            "pattern": pattern,
                            "replacement": replacement
                        }]
                    })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": True,
            "input": text,
            "output": output_text,
            "converged": True,
            "passes": 1 if applied_rules else 0,
            "transformations": applied_rules,
            "traces": traces,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def extract_stems(self, tokens):
        """Extract potential word stems"""
        stems = []
        for token in tokens:
            if len(token) > 3:
                # Simple stem extraction - remove common endings
                if token.endswith('asya'):
                    stems.append(token[:-4])
                elif token.endswith('āya'):
                    stems.append(token[:-3])
                elif token.endswith('am'):
                    stems.append(token[:-2])
                else:
                    stems.append(token)
            else:
                stems.append(token)
        return stems
    
    def extract_suffixes(self, tokens):
        """Extract potential suffixes"""
        suffixes = []
        for token in tokens:
            if token.endswith('asya'):
                suffixes.append('asya (genitive)')
            elif token.endswith('āya'):
                suffixes.append('āya (dative)')
            elif token.endswith('am'):
                suffixes.append('am (accusative)')
        return suffixes
    
    def detect_compounds(self, text):
        """Detect potential compound words"""
        compounds = []
        if '+' in text:
            parts = text.split('+')
            if len(parts) == 2:
                compounds.append({
                    "compound": text.replace(' + ', ''),
                    "parts": [p.strip() for p in parts],
                    "type": "tatpuruṣa"
                })
        return compounds
    
    def get_available_transformations(self, text):
        """Get list of transformations that could apply to this text"""
        available = []
        
        if 'rāma' in text and 'iti' in text:
            available.append("Vowel sandhi: rāma + iti → rāmeti")
        if 'deva' in text and 'indra' in text:
            available.append("Vowel sandhi: deva + indra → devendra")
        if '+' in text:
            available.append("Compound formation available")
        if any(word.endswith('a') for word in text.split()):
            available.append("Case inflection available")
            
        return available
    
    def send_json_response(self, data):
        """Send JSON response with CORS headers"""
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()
            
            json_data = json.dumps(data, indent=2, ensure_ascii=False)
            self.wfile.write(json_data.encode('utf-8'))
        except Exception as e:
            print(f"Error sending JSON response: {e}")
            # Connection likely closed by client

def main():
    """Start the server"""
    host = 'localhost'
    port = 8000
    
    try:
        server = HTTPServer((host, port), SanskritAPIHandler)
        print(f"✅ Server started successfully!")
        print(f"🌐 Server running at: http://{host}:{port}")
        print(f"📡 API endpoints available:")
        print(f"   - GET  /health")
        print(f"   - POST /api/process")
        print(f"   - GET  /api/rules")
        print(f"   - POST /api/analyze")
        print(f"\n🔗 Frontend can now connect to: http://{host}:{port}")
        print(f"⏹️  Press Ctrl+C to stop the server")
        print("=" * 50)
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use!")
            print("💡 Try stopping other servers or use a different port")
        else:
            print(f"❌ Server error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()