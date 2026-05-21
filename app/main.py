"""
Ottoman Turkish Transliteration API
=====================================
Run:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health
    POST /transliterate
    POST /transliterate/batch
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import (
    BatchRequest, BatchResponse, BatchResultItem,
    HealthResponse,
    TokenDetail,
    TransliterateRequest, TransliterateResponse,
)
from app.transliterator import OttomanTransliterator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("ottoman_api")

# ── Engine lifecycle ────────────────────────────────────────────────────────
_engine: Optional[OttomanTransliterator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    log.info("Loading OttomanTransliterator (Zeyrek init may take a moment)…")
    _engine = OttomanTransliterator(
        lookup_file=os.environ.get("LOOKUP_FILE", "manual_lookup.tsv"),
        abbrev_file=os.environ.get("ABBREV_FILE", "abbrev_lookup.tsv"),
        historical=os.environ.get("HISTORICAL_ORTHOGRAPHY", "true").lower() == "true",
    )
    log.info("Engine ready.")
    yield
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ottoman Turkish Transliteration API",
    description=(
        "Converts Modern Turkish text into Ottoman Arabic script (حروف عثمانیه) "
        "using morphological analysis (Zeyrek), rule-based allomorph generation, "
        "and English loanword transliteration."
    ),
    version="2.0.5",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Exception handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Helper ─────────────────────────────────────────────────────────────────
def _get_engine() -> OttomanTransliterator:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return _engine


def _result_to_response(
    result,
    include_tokens: bool,
) -> dict:
    tokens = None
    if include_tokens:
        tokens = [
            TokenDetail(
                token   = t["token"],
                ottoman = t["ottoman"],
                source  = t["source"],
                debug   = t.get("debug", ""),
            )
            for t in result.tokens
        ]
    return dict(
        turkish    = result.turkish,
        ottoman    = result.ottoman,
        confidence = result.confidence,
        tokens     = tokens,
    )


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Liveness check."""
    return HealthResponse()


@app.post(
    "/transliterate",
    response_model=TransliterateResponse,
    tags=["transliteration"],
    summary="Transliterate a single text",
)
async def transliterate(body: TransliterateRequest):
    """
    Convert a Modern Turkish sentence (or paragraph) into Ottoman Arabic script.

    - **text**: Turkish input (max 5 000 characters)
    - **historical**: enable Ottoman orthographic overrides (default `true`)
    - **include_tokens**: return per-token analysis detail (default `false`)
    """
    engine = _get_engine()
    # Honour per-request historical flag by temporarily swapping the setting
    original = engine.historical
    engine.historical = body.historical
    try:
        result = engine.transliterate(body.text)
    finally:
        engine.historical = original

    return TransliterateResponse(**_result_to_response(result, body.include_tokens))


@app.post(
    "/transliterate/batch",
    response_model=BatchResponse,
    tags=["transliteration"],
    summary="Transliterate a batch of texts (max 100)",
)
async def transliterate_batch(body: BatchRequest):
    """
    Process up to 100 sentences in a single request.

    Each item may carry an optional `id` field that is echoed back in the
    response so callers can correlate results to inputs.
    """
    engine = _get_engine()
    original = engine.historical
    engine.historical = body.historical
    results: list[BatchResultItem] = []
    try:
        for item in body.items:
            result = engine.transliterate(item.text)
            row    = _result_to_response(result, body.include_tokens)
            results.append(BatchResultItem(id=item.id, **row))
    finally:
        engine.historical = original

    return BatchResponse(results=results)
