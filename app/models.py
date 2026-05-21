"""Pydantic schemas for the Ottoman transliteration API."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class TransliterateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Modern Turkish text to transliterate")
    historical: bool = Field(True, description="Use historical Ottoman orthography overrides")
    include_tokens: bool = Field(False, description="Include per-token detail in the response")


class TokenDetail(BaseModel):
    token:   str = Field(..., description="Original token")
    ottoman: str = Field(..., description="Ottoman Arabic-script equivalent")
    source:  str = Field(..., description="How the token was resolved: exact | override | tags | english | auto | missing | punct")
    debug:   str = Field("",  description="Debug info: lemma :: tags :: surface")


class TransliterateResponse(BaseModel):
    turkish:    str           = Field(..., description="Original Turkish input")
    ottoman:    str           = Field(..., description="Ottoman Arabic-script output")
    confidence: float         = Field(..., ge=0.0, le=1.0, description="Mean token-level confidence [0–1]")
    tokens:     Optional[list[TokenDetail]] = Field(None, description="Per-token detail (only when include_tokens=true)")


class BatchItem(BaseModel):
    id:   Optional[str] = Field(None, description="Optional caller-supplied identifier")
    text: str           = Field(..., min_length=1, max_length=5000)


class BatchRequest(BaseModel):
    items:          list[BatchItem] = Field(..., min_length=1, max_length=100)
    historical:     bool = Field(True)
    include_tokens: bool = Field(False)


class BatchResultItem(BaseModel):
    id:         Optional[str]
    turkish:    str
    ottoman:    str
    confidence: float
    tokens:     Optional[list[TokenDetail]] = None


class BatchResponse(BaseModel):
    results: list[BatchResultItem]


class HealthResponse(BaseModel):
    status:  str = "ok"
    engine:  str = "OttomanTransliterator"
    version: str = "2.0.5"
