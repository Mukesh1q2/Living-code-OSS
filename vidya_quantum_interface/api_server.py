from fastapi import FastAPI
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

# Additional Kubernetes-friendly health endpoints
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/readyz")
async def readyz():
    # TODO: add deeper readiness checks (models, external services)
    return {"ready": True}

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
