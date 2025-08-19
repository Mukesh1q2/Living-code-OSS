# Roadmap (first 4–6 weeks)

## Week 1–2
- CI: lint + typecheck + tests across Python 3.11/3.12, Node 20.
- Security: SECURITY.md, Dependabot, CodeQL (enable in GitHub UI).
- Repo consolidation: canonical backend/frontend; move legacy to ARCHIVE/.

## Week 3–4
- Observability: `/healthz`, `/readyz`, structured logs, metrics.
- Test fixtures: small Sanskrit corpus with expected outputs; coverage ≥70%.
- Docker: `docker compose up` runs backend + (optional) frontend.

## Week 5–6
- Performance: cache morphology results; seed & log determinism for LLM flows.
- Docs: task‑based examples; architecture diagram; troubleshooting guide.

