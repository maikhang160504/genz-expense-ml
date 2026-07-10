"""High-level OCR service. Real pipeline (PaddleOCR + VietOCR) with mock fallback."""
from __future__ import annotations

import time
from typing import Any

from app.adapters import expense_ocr_nlu as adapter
from app.adapters.mock_pipeline import mock_ocr_from_text
from app.core.config import get_settings
from app.core.exceptions import OCRProcessingError
from app.core.logging import get_logger
from app.schemas.ocr import OCRResponse


logger = get_logger(__name__)


MOCK_RECEIPT_TEXT = """
SIEU THI VINMART
KEM TRA SUA TRAN CHAU 35.000
CAFE SUA DA 25.000
BANH MI THIT 20.000
TONG TIEN 80.000
""".strip()


class OCRService:
    def __init__(self) -> None:
        self._tried_load = False
        self._real_available = False

    def try_load(self) -> bool:
        if self._tried_load:
            return self._real_available
        self._tried_load = True
        if not get_settings().use_real_ocr:
            logger.info("USE_REAL_OCR=false -> mock OCR only.")
            return False
        self._real_available = adapter.load_real_ocr_safe()
        if not self._real_available:
            logger.warning("Real OCR not loadable: %s", adapter.get_ocr_error())
        return self._real_available

    def reload(self) -> bool:
        """Hot-reload OCR weights (VietOCR + PICK KIE) after deploy."""
        self._tried_load = False
        self._real_available = False
        if not get_settings().use_real_ocr:
            logger.info("USE_REAL_OCR=false — skip OCR reload.")
            return False
        self._tried_load = True
        self._real_available = adapter.reload_ocr()
        if not self._real_available:
            logger.warning("OCR reload failed: %s", adapter.get_ocr_error())
        else:
            logger.info("OCR pipeline reloaded from disk.")
        return self._real_available

    def infer_image(self, image_bytes: bytes, filename: str | None = None) -> OCRResponse:
        start = time.perf_counter()
        try:
            if self._real_available or self.try_load():
                try:
                    payload = adapter.run_real_ocr(image_bytes, filename)
                    payload["latency_ms"] = int((time.perf_counter() - start) * 1000)
                    return OCRResponse.model_validate(payload)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Real OCR failed, falling back to mock: %s", exc)
            payload = mock_ocr_from_text(MOCK_RECEIPT_TEXT)
            payload["latency_ms"] = int((time.perf_counter() - start) * 1000)
            return OCRResponse.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise OCRProcessingError(f"OCR processing failed: {exc}") from exc

    def infer_text(self, raw_text: str) -> OCRResponse:
        start = time.perf_counter()
        payload = mock_ocr_from_text(raw_text)
        payload["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return OCRResponse.model_validate(payload)


_singleton: OCRService | None = None


def get_ocr_service() -> OCRService:
    global _singleton
    if _singleton is None:
        _singleton = OCRService()
    return _singleton
