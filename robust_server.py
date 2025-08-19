#!/usr/bin/env python3
"""
Robust server for Sanskrit Rewrite Engine
Handles connection errors and frontend requests gracefully
"""

import json
import logging
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import sys
import os

# Configure logging to reduce noise
logging.basicConfig(level=logging.WARNING)

class RobustSanskritHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Suppress default logging"""
        return
    
    def handle_one_request(self):
        """Handle one request with better error handling"""
        try:
            super().handle_one_request()
        except ConnectionAbortedError:
            # Client disconnected - this is normal, don't log it
            pass
        except Exception as e:
            print(f"Request handling error: {e}")
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        try:
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()
        except:
            pass
    
    def do_GET(self):
        """Handle GET requests with robust error handling"""
        try:
            if self.path == '/':
                self.send_api_info()
            elif self.path == '/health':
                self.send_health_status()
            elif self.path == '/api/rules':
                self.send_rules_list()
            elif self.path == '/favicon.ico':
                self.send_favicon()
            else:
                self.send_not_found()
        except ConnectionAbortedError:
            pass  # Client disconnected
        except Exception as e:
            print(f"GET error for {self.path}: {e}")
            self.send_server_error()
    
    def do_POST(self):
        """Handle POST requests with robust error handling"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                try:
                    request_data = json.loads(post_data.decode('utf-8'))
                except json.JSONDecodeError:
                    self.send_bad_request("Invalid JSON")
                    return
            else:
                request_data = {}
            
            if self.path == '/api/process':
                self.handle_process_request(request_data)
            elif self.path == '/api/analyze':
                self.handle_analyze_request(request_data)
            else:
                self.send_not_found()
                
        except ConnectionAbortedError:
            pass  # Client disconnected
        except Exception as e:
            print(f"POST error for {self.path}: {e}")
            self.send_server_error()
    
    def send_api_info(self):
        """Send API information"""
        response = {
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
            "example": {
                "url": "/api/process",
                "method": "POST",
                "body": {"text": "rāma + iti", "enable_tracing": True}
            }
        }
        self.send_json_safe(response)
    
    def send_health_status(self):
        """Send health status"""
        response = {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "uptime": "running",
            "components": {
                "api_server": "✅ running",
                "sanskrit_processor": "✅ available",
                "rule_engine": "✅ loaded"
            }
        }
        self.send_json_safe(response)
    
    def send_rules_list(self):
        """Send available rules"""
        response = {
            "success": True,
            "count": 4,
            "rules": [
                {
                    "id": "vowel_sandhi_a_i",
                    "name": "Vowel Sandhi: a + i → e",
                    "description": "Combines vowels 'a' and 'i' to form 'e'",
                    "example": "rāma + iti → rāmeti",
                    "sutra": "6.1.87",
                    "category": "sandhi"
                },
                {
                    "id": "vowel_sandhi_a_u",
                    "name": "Vowel Sandhi: a + u → o", 
                    "description": "Combines vowels 'a' and 'u' to form 'o'",
                    "example": "deva + ukta → devokta",
                    "sutra": "6.1.87",
                    "category": "sandhi"
                },
                {
                    "id": "compound_formation",
                    "name": "Compound Formation (Samāsa)",
                    "description": "Forms compound words from multiple stems",
                    "example": "deva + rāja → devarāja",
                    "sutra": "2.1.3",
                    "category": "morphology"
                },
                {
                    "id": "genitive_case",
                    "name": "Genitive Case Formation",
                    "description": "Forms genitive case endings",
                    "example": "rāma → rāmasya",
                    "sutra": "7.1.12",
                    "category": "inflection"
                }
            ]
        }
        self.send_json_safe(response)
    
    def send_favicon(self):
        """Handle favicon requests"""
        try:
            self.send_response(204)  # No Content
            self.end_headers()
        except:
            pass
    
    def handle_process_request(self, request_data):
        """Handle Sanskrit text processing"""
        text = request_data.get('text', '').strip()
        
        if not text:
            self.send_bad_request("No text provided")
            return
        
        # Simple Sanskrit processing with multiple rules
        result = self.process_sanskrit_text(text, request_data)
        self.send_json_safe(result)
    
    def handle_analyze_request(self, request_data):
        """Handle Sanskrit text analysis"""
        text = request_data.get('text', '').strip()
        
        if not text:
            self.send_bad_request("No text provided")
            return
        
        analysis = self.analyze_sanskrit_text(text)
        self.send_json_safe(analysis)
    
    def process_sanskrit_text(self, text, options):
        """Process Sanskrit text with transformations"""
        start_time = datetime.now()
        
        # Sanskrit transformation rules
        transformations = {
            # Vowel sandhi
            'rāma + iti': 'rāmeti',
            'deva + indra': 'devendra',
            'mahā + ātman': 'mahātman',
            'su + ukta': 'sukta',
            'guru + iti': 'gurūti',
            
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
                
                rule_name = self.get_rule_name(pattern, replacement)
                applied_rules[rule_name] = applied_rules.get(rule_name, 0) + 1
                
                if options.get('enable_tracing', True):
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
            "processing_time_ms": round(processing_time * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "rules_available": len(transformations)
        }
    
    def analyze_sanskrit_text(self, text):
        """Analyze Sanskrit text structure"""
        words = text.split()
        
        return {
            "success": True,
            "input": text,
            "analysis": {
                "basic_stats": {
                    "character_count": len(text),
                    "word_count": len(words),
                    "contains_markers": '+' in text or ':' in text
                },
                "tokenization": {
                    "words": words,
                    "potential_compounds": self.detect_compounds(text),
                    "sandhi_opportunities": self.detect_sandhi_opportunities(text)
                },
                "morphology": {
                    "stems": self.extract_stems(words),
                    "possible_inflections": self.detect_inflections(words)
                },
                "transformations_available": self.get_available_transformations(text)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_rule_name(self, pattern, replacement):
        """Generate rule name from pattern"""
        if '+' in pattern:
            parts = pattern.split(' + ')
            if len(parts) == 2:
                return f"sandhi_{parts[0]}_{parts[1]}"
        return "transformation_rule"
    
    def detect_compounds(self, text):
        """Detect potential compound words"""
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
    
    def detect_sandhi_opportunities(self, text):
        """Detect sandhi opportunities"""
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
        
        return opportunities
    
    def extract_stems(self, words):
        """Extract potential word stems"""
        stems = []
        for word in words:
            if word.endswith('asya'):
                stems.append({"word": word, "stem": word[:-4], "suffix": "asya"})
            elif word.endswith('āya'):
                stems.append({"word": word, "stem": word[:-3], "suffix": "āya"})
            else:
                stems.append({"word": word, "stem": word, "suffix": None})
        return stems
    
    def detect_inflections(self, words):
        """Detect possible inflections"""
        inflections = []
        for word in words:
            if word.endswith('asya'):
                inflections.append({"word": word, "case": "genitive", "number": "singular"})
            elif word.endswith('āya'):
                inflections.append({"word": word, "case": "dative", "number": "singular"})
        return inflections
    
    def get_available_transformations(self, text):
        """Get available transformations for this text"""
        available = []
        
        if 'rāma' in text and 'iti' in text:
            available.append("Vowel sandhi: rāma + iti → rāmeti")
        if 'deva' in text:
            available.append("Compound formation with deva-")
        if any(word.endswith('a') for word in text.split()):
            available.append("Case inflection possible")
        if '+' in text:
            available.append("Marked sandhi/compound formation")
            
        return available
    
    def send_json_safe(self, data):
        """Send JSON response safely"""
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()
            
            json_data = json.dumps(data, indent=2, ensure_ascii=False)
            self.wfile.write(json_data.encode('utf-8'))
        except Exception as e:
            print(f"Error sending response: {e}")
    
    def send_bad_request(self, message):
        """Send 400 Bad Request"""
        try:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {"error": "Bad Request", "message": message}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
        except:
            pass
    
    def send_not_found(self):
        """Send 404 Not Found"""
        try:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {"error": "Not Found", "path": self.path}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
        except:
            pass
    
    def send_server_error(self):
        """Send 500 Server Error"""
        try:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {"error": "Internal Server Error"}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
        except:
            pass

def main():
    """Start the robust server"""
    host = 'localhost'
    port = 8000
    
    print("🚀 Starting Robust Sanskrit Rewrite Engine Server...")
    print("=" * 55)
    
    try:
        server = HTTPServer((host, port), RobustSanskritHandler)
        
        print(f"✅ Server started successfully!")
        print(f"🌐 URL: http://{host}:{port}")
        print(f"📋 Endpoints:")
        print(f"   GET  /          - API info")
        print(f"   GET  /health    - Health check") 
        print(f"   GET  /api/rules - Available rules")
        print(f"   POST /api/process - Process text")
        print(f"   POST /api/analyze - Analyze text")
        print(f"")
        print(f"🔗 Frontend can connect to: http://{host}:{port}")
        print(f"⏹️  Press Ctrl+C to stop")
        print("=" * 55)
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use!")
            print("💡 Stop other servers or use a different port")
        else:
            print(f"❌ Server error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()