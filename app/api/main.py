import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="Agentic AI Research & Decision System",
    description=(
        "API for evidence-based research using RAG, "
        "verification, and decision-making."
    ),
    version="1.0.0",
)

# Comma-separated list of allowed frontend origins, e.g.
# "http://localhost:5173,https://your-frontend.vercel.app"
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Agentic AI Research & Decision System",
    }


@app.post("/research")
def research(request: ResearchRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Research question cannot be empty.",
        )

    # Imported lazily so that /health (and the app itself) can
    # start and be tested without pulling in the full RAG stack
    # (FAISS, sentence-transformers, torch). Those are only
    # needed once an actual research question is submitted.
    from app.core.orchestrator import run_research_system

    try:
        result = run_research_system(question)
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
