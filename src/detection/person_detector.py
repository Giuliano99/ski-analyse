"""Lokale Personenerkennung fuer das Athletentracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonDetection:
    bbox: tuple[float, float, float, float]
    confidence: float


class LocalPersonDetector:
    """Kleine YOLO-Personenerkennung; Videoframes bleiben lokal."""

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        *,
        confidence: float = 0.05,
        image_size: int = 1280,
    ) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("confidence muss zwischen 0 und 1 liegen")
        if image_size <= 0:
            raise ValueError("image_size muss positiv sein")

        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._image_size = image_size

    def detect(self, frame) -> list[PersonDetection]:
        result = self._model.predict(
            frame,
            classes=[0],
            conf=self._confidence,
            imgsz=self._image_size,
            verbose=False,
        )[0]
        detections: list[PersonDetection] = []
        for xyxy, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            detections.append(
                PersonDetection(
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    confidence=float(confidence),
                )
            )
        return detections
