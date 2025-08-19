from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI(title="Vidya Quantum Interface")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Vidya Quantum Interface API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Kubernetes-friendly health endpoints
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/readyz")
async def readyz():
    # TODO: add deeper readiness checks (models, external services)
    return {"ready": True}

# Minimal favicon to prevent 404s in browsers
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.post("/api/sanskrit/analyze")
async def analyze_sanskrit(data: dict):
    text = data.get("text", "")
    return {
        "input": text,
        "analysis": {
            "words": text.split(),
            "word_count": len(text.split()),
            "character_count": len(text)
        }
    }

@app.post("/api/process")
async def process_text(data: dict):
    text = data.get("text", "")
    # Simulate processing pipeline
    output_text = f"Processed: {text}"
    transformations_applied = [
        "tokenize",
        "analyze_morphology",
        "apply_rules",
    ]
    return {
        "success": True,
        "output_text": output_text,
        "transformations_applied": transformations_applied,
        "timestamp": datetime.utcnow().isoformat(),
    }

# Minimal WebSocket endpoint for live messaging
active_ws_clients = set()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print(f"[WS] client connected @ {datetime.utcnow().isoformat()}")
    active_ws_clients.add(ws)
    try:
        # Notify client of connection
        await ws.send_json({"type": "connection_established", "payload": {"ts": datetime.utcnow().isoformat()}})
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            payload = msg.get("payload", {})
            print(f"[WS] rx type={mtype} payload_keys={list(payload.keys())}")

            if mtype == "ping":
                await ws.send_json({"type": "pong", "payload": {"ts": datetime.utcnow().isoformat()}})
            elif mtype == "user_input":
                text = payload.get("input", "")
                await ws.send_json({
                    "type": "vidya_response",
                    "payload": {"text": f"Vidya (live): You said: {text}", "sanskrit_analysis": None},
                    "timestamp": datetime.utcnow().isoformat(),
                })
            elif mtype == "analyze_sanskrit":
                text = payload.get("text", "")
                await ws.send_json({
                    "type": "sanskrit_analysis",
                    "payload": {"tokens": text.split(), "rulesFired": ["example_rule"]},
                    "timestamp": datetime.utcnow().isoformat(),
                })
            elif mtype == "process_text":
                text = payload.get("text", "")
                await ws.send_json({"type": "sanskrit_processing_update", "payload": {"stage": "start"}})
                await ws.send_json({
                    "type": "vidya_response",
                    "payload": {"text": f"Processed (live): {text}", "sanskrit_analysis": None},
                    "timestamp": datetime.utcnow().isoformat(),
                })
                await ws.send_json({"type": "processing_complete", "payload": {}})
            else:
                await ws.send_json({"type": "error", "payload": {"error": f"Unknown message type: {mtype}"}})
    except WebSocketDisconnect:
        print(f"[WS] client disconnected @ {datetime.utcnow().isoformat()}")
    finally:
        active_ws_clients.discard(ws)
        print(f"[WS] active clients: {len(active_ws_clients)}")
