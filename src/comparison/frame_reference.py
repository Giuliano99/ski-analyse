"""Bestaetigte Referenzframes fuer automatische Torereignisse auswerten."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GateFrameReference:
    gate_id: int
    frame_number: int


@dataclass(frozen=True)
class FrameReferenceDocument:
    video_name: str
    tolerance_frames: int
    crossings: list[GateFrameReference]

    def __post_init__(self) -> None:
        if self.tolerance_frames < 0:
            raise ValueError("tolerance_frames darf nicht negativ sein")
        gate_ids = [item.gate_id for item in self.crossings]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("Referenz enthaelt doppelte gate_id")


@dataclass(frozen=True)
class GateFrameEvaluation:
    gate_id: int
    reference_frame: int
    predicted_frame: int | None
    frame_error: int | None
    within_tolerance: bool


def load_frame_references(path: str | Path) -> FrameReferenceDocument:
    with Path(path).open(encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file)
    return FrameReferenceDocument(
        video_name=str(data["video_name"]),
        tolerance_frames=int(data["tolerance_frames"]),
        crossings=[
            GateFrameReference(
                gate_id=int(item["gate_id"]),
                frame_number=int(item["frame_number"]),
            )
            for item in data["crossings"]
        ],
    )


def evaluate_predicted_frames(
    references: FrameReferenceDocument,
    predictions: dict[int, int],
) -> list[GateFrameEvaluation]:
    evaluations = []
    for reference in references.crossings:
        predicted = predictions.get(reference.gate_id)
        error = None if predicted is None else predicted - reference.frame_number
        evaluations.append(
            GateFrameEvaluation(
                gate_id=reference.gate_id,
                reference_frame=reference.frame_number,
                predicted_frame=predicted,
                frame_error=error,
                within_tolerance=(
                    error is not None and abs(error) <= references.tolerance_frames
                ),
            )
        )
    return evaluations
