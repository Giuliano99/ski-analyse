"""Datenmodell und JSON Speicherung manueller Tordurchfahrten."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.timing.geometry import Point, moving_gate_crossing_fraction
from src.timing.lap_time import GateCrossingObservation


@dataclass(frozen=True)
class FrameAnnotation:
    frame_number: int
    timestamp_s: float
    athlete_pos: Point
    gate_left: Point
    gate_right: Point


@dataclass(frozen=True)
class ManualGateCrossing:
    gate_id: int
    previous: FrameAnnotation
    current: FrameAnnotation
    confidence: float = 1.0
    uncertain: bool = False
    uncertain_reason: str | None = None

    def __post_init__(self) -> None:
        if self.current.frame_number <= self.previous.frame_number:
            raise ValueError("Der aktuelle Frame muss nach dem vorherigen Frame liegen")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence muss zwischen 0 und 1 liegen")


@dataclass
class ManualAnnotationDocument:
    video_name: str
    width: int
    height: int
    fps: float
    target_athlete: str
    crossings: list[ManualGateCrossing] = field(default_factory=list)
    version: int = 1

    def add_crossing(self, crossing: ManualGateCrossing) -> None:
        if any(item.gate_id == crossing.gate_id for item in self.crossings):
            raise ValueError(f"gate_id {crossing.gate_id} ist bereits vorhanden")
        self.crossings.append(crossing)
        self.crossings.sort(key=lambda item: item.gate_id)


def _point(data: dict[str, float]) -> Point:
    return Point(float(data["x"]), float(data["y"]))


def _frame(data: dict[str, object]) -> FrameAnnotation:
    return FrameAnnotation(
        frame_number=int(data["frame_number"]),
        timestamp_s=float(data["timestamp_s"]),
        athlete_pos=_point(data["athlete_pos"]),
        gate_left=_point(data["gate_left"]),
        gate_right=_point(data["gate_right"]),
    )


def load_document(path: str | Path) -> ManualAnnotationDocument:
    source = Path(path)
    with source.open(encoding="utf-8") as annotation_file:
        data = json.load(annotation_file)
    if data.get("version") != 1:
        raise ValueError(f"Nicht unterstuetzte Annotationsversion in {source}")

    document = ManualAnnotationDocument(
        video_name=str(data["video_name"]),
        width=int(data["width"]),
        height=int(data["height"]),
        fps=float(data["fps"]),
        target_athlete=str(data["target_athlete"]),
        version=int(data["version"]),
    )
    for item in data.get("crossings", []):
        document.add_crossing(
            ManualGateCrossing(
                gate_id=int(item["gate_id"]),
                previous=_frame(item["previous"]),
                current=_frame(item["current"]),
                confidence=float(item.get("confidence", 1.0)),
                uncertain=bool(item.get("uncertain", False)),
                uncertain_reason=item.get("uncertain_reason"),
            )
        )
    return document


def save_document(document: ManualAnnotationDocument, path: str | Path) -> Path:
    """Speichert atomar, damit ein Abbruch keine Annotationen zerstoert."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(document), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def to_observations(
    document: ManualAnnotationDocument,
) -> list[GateCrossingObservation]:
    observations = []
    for crossing in document.crossings:
        fraction = moving_gate_crossing_fraction(
            crossing.previous.athlete_pos,
            crossing.current.athlete_pos,
            crossing.previous.gate_left,
            crossing.previous.gate_right,
            crossing.current.gate_left,
            crossing.current.gate_right,
        )
        uncertain = crossing.uncertain
        uncertain_reason = crossing.uncertain_reason
        fraction_override = None
        if fraction is None:
            fraction_override = 0.5
            uncertain = True
            fallback_reason = (
                "Manuell bestaetigte Durchfahrt ohne geometrischen Seitenwechsel; "
                "Zeitpunkt ist die Mitte des markierten Frameintervalls"
            )
            uncertain_reason = (
                f"{uncertain_reason}; {fallback_reason}"
                if uncertain_reason
                else fallback_reason
            )
        observations.append(
            GateCrossingObservation(
                gate_id=crossing.gate_id,
                prev_frame_number=crossing.previous.frame_number,
                curr_frame_number=crossing.current.frame_number,
                prev_timestamp_s=crossing.previous.timestamp_s,
                curr_timestamp_s=crossing.current.timestamp_s,
                prev_athlete_pos=crossing.previous.athlete_pos,
                curr_athlete_pos=crossing.current.athlete_pos,
                prev_gate_left=crossing.previous.gate_left,
                prev_gate_right=crossing.previous.gate_right,
                curr_gate_left=crossing.current.gate_left,
                curr_gate_right=crossing.current.gate_right,
                confidence=crossing.confidence,
                uncertain=uncertain,
                uncertain_reason=uncertain_reason,
                crossing_fraction_override=fraction_override,
            )
        )
    return observations
