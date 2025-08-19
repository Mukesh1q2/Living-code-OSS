# Architecture (proposed consolidation)

```
+--------------------------------------------------------------+
|                         Web Frontend                         |
|             (React/Vite, optional WebGL/3D view)             |
+------------------------^------------------^------------------+
                         |                  |
                REST/WS API          Static assets
                         |                  |
+------------------------+------------------+------------------+
|                   FastAPI Backend (Python)                   |
|  - API routers (analysis, morphology, etymology)             |
|  - Pydantic models & validation                              |
|  - Observability: logging, /healthz, /readyz, metrics        |
|  - Integration with LLMs / Sanskrit rule engine              |
+------------------------+------------------+------------------+
                         |                  |
                 Data stores           External models/LLMs
```
## Notes
- Canonical backend package: `vidya_quantum_interface/` (suggested).
- Canonical frontend directory: `frontend/` (or `vidya-frontend/` if already established).
- Move old/prototype code into `ARCHIVE/` to avoid import drift.

