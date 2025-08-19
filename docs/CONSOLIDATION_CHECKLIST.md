# Consolidation Checklist

- [ ] Choose canonical backend dir (e.g., `vidya_quantum_interface/`). Remove/Archive duplicates.
- [ ] Choose canonical frontend dir (`frontend/` or `vidya-frontend/`). Remove/Archive duplicates.
- [ ] Update imports and scripts to point to canonical names.
- [ ] Update README and docs to reflect the new layout.
- [ ] Add `/healthz` and `/readyz` endpoints and basic metrics in the backend.
- [ ] Ensure `make validate` passes locally and in CI.

