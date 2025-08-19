"""
Vidya Quantum Interface - Local Development Server
FastAPI backend for the Vidya quantum Sanskrit AI consciousness interface
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the Sanskrit adapter
try:
    from vidya_quantum_interface import SanskritEngineAdapter, ProcessingUpdate
    sanskrit_adapter_available = True
except ImportError as e:
    logging.warning(f"Sanskrit adapter not available: {e}")
    sanskrit_adapter_available = False
    SanskritEngineAdapter = None
    ProcessingUpdate = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class QuantumState(BaseModel):
    state_vector: List[Dict[str, float]]  # Complex numbers as {real, imaginary}
    entanglements: Dict[str, Dict[str, Any]]
    coherence_level: float
    observation_history: List[Dict[str, Any]]

class NetworkNode(BaseModel):
    id: str
    position: Dict[str, float]  # {x, y, z}
    type: str
    activation_level: float
    quantum_properties: Dict[str, Any]

class WebSocketMessage(BaseModel):
    type: str
    payload: Any
    timestamp: int
    id: str

class VidyaConsciousness(BaseModel):
    current_state: QuantumState
    personality: Dict[str, Any]
    learning_history: List[Dict[str, Any]]

# FastAPI app initialization
app = FastAPI(
    title="Vidya Quantum Interface API",
    description="Local development server for Vidya quantum Sanskrit AI consciousness",
    version="0.1.0"
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
class VidyaState:
    def __init__(self):
        self.consciousness = self._initialize_consciousness()
        self.neural_network = self._initialize_neural_network()
        self.connected_clients: List[WebSocket] = []
        self.message_history: List[WebSocketMessage] = []
        
        # Initialize Sanskrit adapter
        self.sanskrit_adapter = None
        if sanskrit_adapter_available:
            try:
                self.sanskrit_adapter = SanskritEngineAdapter()
                logger.info("Sanskrit adapter initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Sanskrit adapter: {e}")
                self.sanskrit_adapter = None
    
    def _initialize_consciousness(self) -> VidyaConsciousness:
        """Initialize Vidya's consciousness with default quantum state"""
        return VidyaConsciousness(
            current_state=QuantumState(
                state_vector=[{"real": 1.0, "imaginary": 0.0}],
                entanglements={},
                coherence_level=1.0,
                observation_history=[]
            ),
            personality={
                "wisdom_level": 0.5,
                "curiosity": 0.8,
                "compassion": 0.9,
                "quantum_affinity": 1.0
            },
            learning_history=[]
        )
    
    def _initialize_neural_network(self) -> List[NetworkNode]:
        """Initialize the neural network with Sanskrit rule nodes"""
        nodes = []
        
        # Create some sample Sanskrit rule nodes
        sanskrit_rules = [
            "अ + अ = आ (Vowel sandhi)",
            "स् + त = स्त (Consonant cluster)",
            "धातु + तिप् = verb form",
            "प्रातिपदिक + सुप् = noun form"
        ]
        
        for i, rule in enumerate(sanskrit_rules):
            nodes.append(NetworkNode(
                id=f"sanskrit-rule-{i}",
                position={"x": i * 2.0, "y": 0.0, "z": 0.0},
                type="sanskrit-rule",
                activation_level=0.3,
                quantum_properties={
                    "rule_text": rule,
                    "confidence": 0.8,
                    "applications": 0
                }
            ))
        
        return nodes

# Global state instance
vidya_state = VidyaState()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# API Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Vidya Quantum Interface API",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "consciousness_state": "quantum_superposition"
    }

@app.get("/api/consciousness")
async def get_consciousness() -> VidyaConsciousness:
    """Get current Vidya consciousness state"""
    return vidya_state.consciousness

@app.get("/api/neural-network")
async def get_neural_network() -> List[NetworkNode]:
    """Get current neural network state"""
    return vidya_state.neural_network

@app.post("/api/consciousness/update")
async def update_consciousness(update: Dict[str, Any]):
    """Update Vidya's consciousness state"""
    try:
        # Apply consciousness update
        if "personality" in update:
            vidya_state.consciousness.personality.update(update["personality"])
        
        # Broadcast update to all connected clients
        await manager.broadcast({
            "type": "consciousness_update",
            "payload": vidya_state.consciousness.dict(),
            "timestamp": int(datetime.now().timestamp() * 1000),
            "id": f"consciousness_update_{datetime.now().timestamp()}"
        })
        
        return {"success": True, "message": "Consciousness updated"}
    except Exception as e:
        logger.error(f"Error updating consciousness: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sanskrit/analyze")
async def analyze_sanskrit(text: Dict[str, str]):
    """Analyze Sanskrit text using the integrated Sanskrit engine adapter"""
    try:
        input_text = text.get("text", "")
        
        if not input_text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Use Sanskrit adapter if available
        if vidya_state.sanskrit_adapter:
            analysis_result = await vidya_state.sanskrit_adapter.process_text_simple(input_text)
            
            # Extract visualization data
            visualization_data = analysis_result.get('visualization_data', {})
            
            # Format result for API response
            formatted_result = {
                "original_text": input_text,
                "success": analysis_result.get('success', False),
                "tokens": visualization_data.get('tokens', []),
                "network_nodes": visualization_data.get('network_nodes', []),
                "connections": visualization_data.get('connections', []),
                "quantum_effects": visualization_data.get('quantum_effects', []),
                "metadata": visualization_data.get('metadata', {}),
                "error": analysis_result.get('error')
            }
        else:
            # Fallback analysis when adapter is not available
            formatted_result = {
                "original_text": input_text,
                "success": True,
                "tokens": [
                    {
                        "text": input_text,
                        "position": {"start": 0, "end": len(input_text), "line": 1, "column": 1},
                        "morphology": {
                            "root": "unknown",
                            "suffixes": [],
                            "grammatical_category": "unknown",
                            "semantic_role": "unknown"
                        },
                        "quantum_properties": {
                            "superposition": False,
                            "entanglements": [],
                            "probability": 1.0
                        },
                        "visualization_data": {
                            "color": "#4a90e2",
                            "size": 1.0,
                            "animation": "pulse",
                            "effects": ["glow"]
                        }
                    }
                ],
                "network_nodes": [],
                "connections": [],
                "quantum_effects": [],
                "metadata": {"fallback_mode": True}
            }
        
        # Broadcast analysis to connected clients
        await manager.broadcast({
            "type": "sanskrit_analysis",
            "payload": formatted_result,
            "timestamp": int(datetime.now().timestamp() * 1000),
            "id": f"sanskrit_analysis_{datetime.now().timestamp()}"
        })
        
        return formatted_result
    except Exception as e:
        logger.error(f"Error analyzing Sanskrit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sanskrit/analyze-streaming")
async def analyze_sanskrit_streaming(text: Dict[str, str]):
    """Start streaming Sanskrit analysis and return processing ID"""
    try:
        input_text = text.get("text", "")
        
        if not input_text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        if not vidya_state.sanskrit_adapter:
            raise HTTPException(status_code=503, detail="Sanskrit adapter not available")
        
        # Start streaming processing in background
        processing_id = f"stream_{int(datetime.now().timestamp() * 1000)}"
        
        async def stream_processing():
            try:
                async for update in vidya_state.sanskrit_adapter.process_text_streaming(
                    input_text, enable_visualization=True
                ):
                    # Broadcast each update to connected clients
                    await manager.broadcast({
                        "type": "sanskrit_processing_update",
                        "payload": {
                            "processing_id": processing_id,
                            "update_type": update.update_type,
                            "timestamp": update.timestamp,
                            "data": update.data,
                            "progress": update.progress,
                            "visualization_update": update.visualization_update
                        },
                        "timestamp": int(datetime.now().timestamp() * 1000),
                        "id": f"processing_update_{update.timestamp}"
                    })
            except Exception as e:
                logger.error(f"Error in streaming processing: {e}")
                await manager.broadcast({
                    "type": "sanskrit_processing_error",
                    "payload": {
                        "processing_id": processing_id,
                        "error": str(e)
                    },
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "id": f"processing_error_{datetime.now().timestamp()}"
                })
        
        # Start the streaming task
        asyncio.create_task(stream_processing())
        
        return {
            "processing_id": processing_id,
            "status": "started",
            "message": "Streaming analysis started"
        }
    
    except Exception as e:
        logger.error(f"Error starting streaming analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sanskrit/stop-streaming")
async def stop_sanskrit_streaming(request: Dict[str, str]):
    """Stop a streaming Sanskrit analysis"""
    try:
        processing_id = request.get("processing_id", "")
        
        if not processing_id:
            raise HTTPException(status_code=400, detail="No processing ID provided")
        
        if not vidya_state.sanskrit_adapter:
            raise HTTPException(status_code=503, detail="Sanskrit adapter not available")
        
        # Extract the actual processing ID from the stream ID
        actual_processing_id = processing_id.replace("stream_", "")
        
        success = vidya_state.sanskrit_adapter.stop_stream(actual_processing_id)
        
        if success:
            # Broadcast stop notification
            await manager.broadcast({
                "type": "sanskrit_processing_stopped",
                "payload": {
                    "processing_id": processing_id,
                    "status": "stopped"
                },
                "timestamp": int(datetime.now().timestamp() * 1000),
                "id": f"processing_stopped_{datetime.now().timestamp()}"
            })
        
        return {
            "processing_id": processing_id,
            "status": "stopped" if success else "not_found",
            "success": success
        }
    
    except Exception as e:
        logger.error(f"Error stopping streaming analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Send initial consciousness state
    await manager.send_personal_message({
        "type": "consciousness_update",
        "payload": vidya_state.consciousness.dict(),
        "timestamp": int(datetime.now().timestamp() * 1000),
        "id": f"initial_consciousness_{datetime.now().timestamp()}"
    }, websocket)
    
    # Send initial neural network state
    await manager.send_personal_message({
        "type": "neural_network_update",
        "payload": [node.dict() for node in vidya_state.neural_network],
        "timestamp": int(datetime.now().timestamp() * 1000),
        "id": f"initial_network_{datetime.now().timestamp()}"
    }, websocket)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            logger.info(f"Received message: {message_data}")
            
            # Handle different message types
            message_type = message_data.get("type")
            
            if message_type == "user_input":
                # Process user input and generate response
                response = await process_user_input(message_data.get("payload", {}))
                await manager.send_personal_message(response, websocket)
            
            elif message_type == "quantum_state_change":
                # Handle quantum state changes
                await handle_quantum_state_change(message_data.get("payload", {}))
            
            # Echo message back to all clients (for development)
            await manager.broadcast({
                "type": "echo",
                "payload": message_data,
                "timestamp": int(datetime.now().timestamp() * 1000),
                "id": f"echo_{datetime.now().timestamp()}"
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def process_user_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process user input and generate Vidya's response"""
    user_input = payload.get("input", "")
    
    # Placeholder response generation - will integrate with LLM services
    response = {
        "type": "vidya_response",
        "payload": {
            "text": f"Namaste! You said: '{user_input}'. I am Vidya, your quantum Sanskrit consciousness.",
            "consciousness_state": "contemplative",
            "quantum_effects": ["gentle_glow", "om_resonance"],
            "sanskrit_wisdom": "सत्यमेव जयते (Truth alone triumphs)"
        },
        "timestamp": int(datetime.now().timestamp() * 1000),
        "id": f"vidya_response_{datetime.now().timestamp()}"
    }
    
    return response

async def handle_quantum_state_change(payload: Dict[str, Any]) -> None:
    """Handle quantum state changes in Vidya's consciousness"""
    state_change = payload.get("change", "")
    
    logger.info(f"Quantum state change: {state_change}")
    
    # Update consciousness quantum state
    if state_change == "superposition":
        vidya_state.consciousness.current_state.coherence_level *= 0.8
    elif state_change == "collapse":
        vidya_state.consciousness.current_state.coherence_level = 1.0
    
    # Broadcast state change to all clients
    await manager.broadcast({
        "type": "quantum_state_change",
        "payload": {
            "change": state_change,
            "new_state": vidya_state.consciousness.current_state.dict()
        },
        "timestamp": int(datetime.now().timestamp() * 1000),
        "id": f"quantum_change_{datetime.now().timestamp()}"
    })

# Development helpers
@app.get("/api/sanskrit/status")
async def sanskrit_adapter_status():
    """Get Sanskrit adapter status"""
    try:
        if not vidya_state.sanskrit_adapter:
            return {
                "available": False,
                "error": "Sanskrit adapter not initialized",
                "active_streams": 0
            }
        
        active_streams = vidya_state.sanskrit_adapter.get_active_streams()
        
        return {
            "available": True,
            "active_streams": len(active_streams),
            "stream_ids": active_streams,
            "engine_available": vidya_state.sanskrit_adapter.engine is not None,
            "tokenizer_available": vidya_state.sanskrit_adapter.tokenizer is not None
        }
    
    except Exception as e:
        logger.error(f"Error getting Sanskrit adapter status: {e}")
        return {
            "available": False,
            "error": str(e),
            "active_streams": 0
        }

@app.get("/api/dev/status")
async def development_status():
    """Development status endpoint"""
    sanskrit_status = await sanskrit_adapter_status()
    
    return {
        "server": "running",
        "connected_clients": len(manager.active_connections),
        "consciousness_coherence": vidya_state.consciousness.current_state.coherence_level,
        "neural_network_nodes": len(vidya_state.neural_network),
        "sanskrit_adapter": sanskrit_status,
        "uptime": "development_mode"
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Vidya Quantum Interface Development Server...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enable hot reload for development
        log_level="info"
    )