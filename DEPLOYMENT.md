# Deployment Guide

## Phase 1: Stable Demo Deployment

Use the deployment-safe fallback first. This proves the frontend, backend, API, CORS, and cloud URL are working.

Environment variables:

```text
USE_EXISTING_PIPELINE=false
USE_QUANTUM_CORE=false
FRONTEND_ORIGIN=*
```

## Phase 2: Existing NLP Pipeline

After the demo deployment works, copy:

```text
integration_pipeline.py
nlp_rag_engine.py
```

into:

```text
backend/app/
```

Then set:

```text
USE_EXISTING_PIPELINE=true
USE_QUANTUM_CORE=false
```

## Phase 3: Full Quantum Core

Only after phase 2 works, copy:

```text
quantum_core.py
```

Then add heavy dependencies carefully:

```text
qiskit
qiskit-algorithms
qiskit-optimization
numpy
pandas
scipy
yfinance
pyarrow
```

Then set:

```text
USE_QUANTUM_CORE=true
```

This phase may need a stronger cloud runtime because Qiskit and market-data downloads can be slow/heavy.

## Cloud choices

### Easiest
- Backend: Render
- Frontend: Vercel

### More cloud/enterprise
- Backend: Google Cloud Run
- Frontend: Vercel or Cloud Run static hosting
- Storage later: Google Cloud Storage or AWS S3
- Logs: Cloud logs
- Secrets: Cloud Secret Manager
