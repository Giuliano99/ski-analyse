"""Roboflow-basierte Erkennung vollstaendiger Riesenslalomtore."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class DetectionModel(Protocol):
    def infer(
        self,
        image: np.ndarray,
        *,
        model_id: str,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class GateDetection:
    """Eine Torbox in Pixelkoordinaten des Originalframes."""

    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (
            round(self.x - self.width / 2),
            round(self.y - self.height / 2),
            round(self.x + self.width / 2),
            round(self.y + self.height / 2),
        )


class GateDetector:
    """Kleine, testbare Huelle um ein Roboflow-Detektionsmodell."""

    def __init__(
        self,
        model: DetectionModel,
        *,
        model_id: str,
        confidence: float = 0.35,
        overlap: float = 0.50,
        class_name: str = "Ski-Gates",
    ):
        if not 0 <= confidence <= 1:
            raise ValueError("confidence muss zwischen 0 und 1 liegen")
        if not 0 <= overlap <= 1:
            raise ValueError("overlap muss zwischen 0 und 1 liegen")

        self.model = model
        self.model_id = model_id
        self.confidence = confidence
        self.overlap = overlap
        self.class_name = class_name

    @classmethod
    def from_roboflow(
        cls,
        *,
        model_id: str,
        confidence: float = 0.35,
        overlap: float = 0.50,
        api_key_env: str = "ROBOFLOW_API_KEY",
    ) -> "GateDetector":
        """Laedt ein Hosted-Modell, ohne den API-Key in Code zu speichern."""

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Umgebungsvariable {api_key_env} ist nicht gesetzt")

        from inference_sdk import InferenceConfiguration, InferenceHTTPClient

        model = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        model.configure(
            InferenceConfiguration(
                confidence_threshold=confidence,
                iou_threshold=overlap,
                api_key_transport="header",
            )
        )
        return cls(
            model,
            model_id=model_id,
            confidence=confidence,
            overlap=overlap,
        )

    def detect(self, image: np.ndarray) -> list[GateDetection]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image muss ein BGR-Farbbild mit drei Kanaelen sein")

        response = self.model.infer(image, model_id=self.model_id)

        detections: list[GateDetection] = []
        for prediction in response.get("predictions", []):
            prediction_class = str(prediction.get("class", ""))
            if prediction_class != self.class_name:
                continue
            prediction_confidence = float(prediction["confidence"])
            if prediction_confidence < self.confidence:
                continue
            detections.append(
                GateDetection(
                    x=float(prediction["x"]),
                    y=float(prediction["y"]),
                    width=float(prediction["width"]),
                    height=float(prediction["height"]),
                    confidence=float(prediction["confidence"]),
                    class_name=prediction_class,
                )
            )
        detections = _remove_contained_components(detections)
        return _non_maximum_suppression(detections, self.overlap)


def _intersection_over_union(first: GateDetection, second: GateDetection) -> float:
    first_x1, first_y1, first_x2, first_y2 = first.xyxy
    second_x1, second_y1, second_x2, second_y2 = second.xyxy
    intersection_width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    return intersection / (first_area + second_area - intersection)


def _area(detection: GateDetection) -> int:
    x1, y1, x2, y2 = detection.xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)


def _contained_fraction(inner: GateDetection, outer: GateDetection) -> float:
    inner_x1, inner_y1, inner_x2, inner_y2 = inner.xyxy
    outer_x1, outer_y1, outer_x2, outer_y2 = outer.xyxy
    intersection_width = max(0, min(inner_x2, outer_x2) - max(inner_x1, outer_x1))
    intersection_height = max(0, min(inner_y2, outer_y2) - max(inner_y1, outer_y1))
    inner_area = _area(inner)
    if inner_area == 0:
        return 0.0
    return intersection_width * intersection_height / inner_area


def _remove_contained_components(
    detections: list[GateDetection],
    threshold: float = 0.80,
) -> list[GateDetection]:
    """Entfernt Panelboxen, die fast vollstaendig in einer Torbox liegen."""

    selected: list[GateDetection] = []
    for detection in sorted(detections, key=_area, reverse=True):
        if any(_contained_fraction(detection, larger) >= threshold for larger in selected):
            continue
        selected.append(detection)
    return selected


def _non_maximum_suppression(
    detections: list[GateDetection],
    overlap: float,
) -> list[GateDetection]:
    remaining = sorted(detections, key=lambda detection: detection.confidence, reverse=True)
    selected: list[GateDetection] = []
    while remaining:
        best = remaining.pop(0)
        selected.append(best)
        remaining = [
            detection
            for detection in remaining
            if _intersection_over_union(best, detection) < overlap
        ]
    return selected
