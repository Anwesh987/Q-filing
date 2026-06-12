import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import OptimizeRequest, OptimizeResponse
from app.services import optimize_from_text, DEFAULT_UNIVERSE

app = FastAPI(
    title="Q-Filing API",
    description="Frontend/cloud API wrapper for Q-Filing compliance + portfolio optimization.",
    version="0.1.0",
)

frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
allowed_origins = ["*"] if frontend_origin == "*" else [o.strip() for o in frontend_origin.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "Q-Filing API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_universe_size": len(DEFAULT_UNIVERSE),
        "use_existing_pipeline": os.getenv("USE_EXISTING_PIPELINE", "false"),
        "use_quantum_core": os.getenv("USE_QUANTUM_CORE", "false"),
    }


@app.post("/optimize/text", response_model=OptimizeResponse)
def optimize_text(payload: OptimizeRequest):
    return optimize_from_text(payload)


@app.post("/optimize/demo", response_model=OptimizeResponse)
def optimize_demo():
    demo_text = """
    PORTFOLIO COMPLIANCE REQUIREMENTS:
    The portfolio must select exactly 10 assets.
    Technology sector exposure must not exceed 4 assets.
    Healthcare sector must represent at least 2 assets.
    Energy sector is capped at 2 assets maximum.
    Financial sector allocation is limited to at most 3 assets.
    """
    request = OptimizeRequest(
        regulatory_text=demo_text,
        horizon_days=90,
        weight_objective="SORTINO",
        use_existing_pipeline=False,
    )
    return optimize_from_text(request)


@app.post("/optimize/file")
async def optimize_file(
    file: UploadFile = File(...),
    horizon_days: int = 90,
    weight_objective: str = "SORTINO",
):
    content = await file.read()

    if file.content_type in {"text/plain", "text/markdown", "application/json"} or file.filename.endswith((".txt", ".md", ".json")):
        text = content.decode("utf-8", errors="ignore")
        return optimize_from_text(
            OptimizeRequest(
                regulatory_text=text,
                horizon_days=horizon_days,
                weight_objective=weight_objective,
            )
        )

    raise HTTPException(
        status_code=415,
        detail="For first deployment, upload .txt/.md/.json or paste text. PDF extraction can be connected later using the existing nlp_rag_engine.py.",
    )
