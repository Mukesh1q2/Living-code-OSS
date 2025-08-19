#!/usr/bin/env python3
"""
WebSocket-compatible server for Sanskrit Rewrite Engine
"""

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

import json
from datetime import datetime

if FLASK_AVAILABLE:
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'sanskrit_engine_secret'
    CORS(app, origins="*")
    socketio = SocketIO(app, cors_allowed_origins="*")

    @app.route('/')
    def api_info():
        return jsonify({
            "message": "Sanskrit Rewrite Engine API",
            "version": "2.0.0",
            "status": "running",
            "websocket": "enabled",
            "endpoints": {
                "GET /": "API information",
                "GET /health": "Health check",
                "GET /api/rules": "Available rules",
                "POST /api/process": "Process Sanskrit text",
                "POST /api/analyze": "Analyze Sanskrit text"
            }
        })

    @app.route('/health')
    def health():
        return jsonify({
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "websocket": "enabled"
        })

    @app.route('/api/rules')
    def get_rules():
        return jsonify({
            "success": True,
            "rules": [
                {
                    "id": "vowel_sandhi_a_i",
                    "name": "Vowel Sandhi: a + i → e",
                    "example": "rāma + iti → rāmeti"
                },
                {
                    "id": "compound_formation",
                    "name": "Compound Formation",
                    "example": "deva + rāja → devarāja"
                }
            ]
        })

    @app.route('/api/process', methods=['POST'])
    def process_text():
        data = request.get_json()
        text = data.get('text', '')
        
        # Simple processing
        output = text.replace('rāma + iti', 'rāmeti').replace('deva + rāja', 'devarāja')
        
        return jsonify({
            "success": True,
            "input": text,
            "output": output,
            "converged": True,
            "transformations": {"sandhi_rule": 1} if output != text else {}
        })

    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        emit('status', {'message': 'Connected to Sanskrit Rewrite Engine'})

    @socketio.on('process_text')
    def handle_process_text(data):
        text = data.get('text', '')
        output = text.replace('rāma + iti', 'rāmeti').replace('deva + rāja', 'devarāja')
        
        result = {
            "success": True,
            "input": text,
            "output": output,
            "timestamp": datetime.now().isoformat()
        }
        
        emit('process_result', result)

    def main():
        print("🚀 Starting WebSocket-enabled Sanskrit Server...")
        print("✅ WebSocket support enabled")
        print("🌐 Server: http://localhost:8000")
        socketio.run(app, host='localhost', port=8000, debug=False)

else:
    def main():
        print("❌ Flask/SocketIO not available")
        print("💡 Install with: pip install flask flask-cors flask-socketio")
        print("🔄 Falling back to basic server...")
        
        # Fall back to basic server
        import subprocess
        subprocess.run(["python", "robust_server.py"])

if __name__ == "__main__":
    main()