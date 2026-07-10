"""Custom exceptions for the AI service."""
from __future__ import annotations


class AIServiceError(Exception):
    """Base error class. HTTP layer maps to 500 unless a subclass overrides."""

    status_code: int = 500
    code: str = "ai_service_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidInputError(AIServiceError):
    status_code = 422
    code = "invalid_input"


class ModelNotLoadedError(AIServiceError):
    status_code = 503
    code = "model_not_loaded"


class OCRProcessingError(AIServiceError):
    status_code = 500
    code = "ocr_processing_error"


class NLUProcessingError(AIServiceError):
    status_code = 500
    code = "nlu_processing_error"
