# Q-Filing Frontend + Cloud Deployment Patch

This patch gives Q-Filing a usable web interface and a deployable FastAPI backend.

## What this includes

```text
backend/
  app/main.py          FastAPI routes
  app/services.py      deployment-safe optimizer wrapper
  app/models.py        Pydantic request/response models
  requirements.txt
  Dockerfile
  render.yaml

frontend/
  src/App.jsx          React dashboard
  src/styles.css       polished UI styling
  package.json
```

## Local run

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000/docs
http://localhost:8000/health
```

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## How to connect the existing team code later

Copy these files from the existing Q-filing branches into:

```text
backend/app/
```

Files to copy:

```text
integration_pipeline.py
nlp_rag_engine.py
quantum_core.py
```

Then set:

```bash
USE_EXISTING_PIPELINE=true
```

Do **not** set `USE_QUANTUM_CORE=true` for first deployment unless all heavy dependencies are installed and tested in cloud.

## Recommended first cloud deployment

Backend:

```text
Render
Root directory: backend
Build command: pip install -r requirements.txt
Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Frontend:

```text
Vercel
Root directory: frontend
Build command: npm run build
Output directory: dist
Environment variable:
VITE_API_BASE_URL=https://your-render-backend-url.onrender.com
```

## What to put on resume

Built the frontend and cloud deployment layer for Q-Filing, an AI + quantum-inspired portfolio compliance optimizer. Developed a React dashboard for compliance text input, extracted constraints, portfolio allocation, and risk metrics. Wrapped the backend with FastAPI, added CORS, health checks, Docker support, and deployment configuration for cloud hosting.
