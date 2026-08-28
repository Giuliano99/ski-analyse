"""
Berechnung von Torzeiten und Abschnittszeiten.

Baut auf den Ergebnissen von geometry.gate_crossed auf. Fuer jedes
erkannte Ereignis werden mindestens folgende Felder gespeichert,
wie im Projektauftrag gefordert:

  gate_id, frame_number, video_timestamp_s, time_since_start_s,
  split_time_s, confidence, uncertain

Der genaue Ueberquerungszeitpunkt zwischen zwei Frames wird per
linearer Interpolation entlang der Bewegungsstrecke geschaetzt,
nicht einfach der Frame mit Ueberquerung genommen. Details werden
in Phase 3 implementiert und mit echten Daten getestet.

Wichtig, siehe Projektauftrag: unsichere oder fehlende Erkennungen
duerfen nicht unbemerkt als korrekt behandelt werden. Jedes Ereignis
muss daher explizit ein uncertain Flag tragen, kein stilles Weglassen.
"""

import csv
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from math import isfinite
from typing import Optional

from src.timing.geometry import Point, moving_gate_crossing_fraction


@dataclass
class GateCrossingEvent:
    gate_id: int
    frame_number: int
    video_timestamp_s: float
    time_since_start_s: float
    split_time_s: Optional[float]
    confidence: float
    uncertain: bool
    uncertain_reason: Optional[str] = None


class StartMode(str, Enum):
    FIRST_GATE = "first_gate"
    MANUAL_FRAME = "manual_frame"


@dataclass(frozen=True)
class TimingConfig:
    start_mode: StartMode
    video_fps: float
    start_gate_id: int = 1
    manual_start_frame: Optional[int] = None

    def __post_init__(self) -> None:
        if not isfinite(self.video_fps) or self.video_fps <= 0:
            raise ValueError("video_fps muss groesser als null sein")
        if self.start_mode == StartMode.MANUAL_FRAME:
            if self.manual_start_frame is None or self.manual_start_frame < 0:
                raise ValueError("manual_frame benoetigt einen gueltigen Startframe")


@dataclass(frozen=True)
class GateCrossingObservation:
    gate_id: int
    prev_frame_number: int
    curr_frame_number: int
    prev_timestamp_s: float
    curr_timestamp_s: float
    prev_athlete_pos: Point
    curr_athlete_pos: Point
    prev_gate_left: Point
    prev_gate_right: Point
    curr_gate_left: Point
    curr_gate_right: Point
    confidence: float = 1.0
    uncertain: bool = False
    uncertain_reason: Optional[str] = None
    crossing_fraction_override: Optional[float] = None


def observation_to_timestamp(observation: GateCrossingObservation) -> float | None:
    """Interpoliert den Durchfahrtszeitpunkt einer manuellen Beobachtung."""
    fraction = observation.crossing_fraction_override
    if fraction is not None and not 0.0 <= fraction <= 1.0:
        raise ValueError("crossing_fraction_override muss zwischen 0 und 1 liegen")
    if fraction is None:
        fraction = moving_gate_crossing_fraction(
            observation.prev_athlete_pos,
            observation.curr_athlete_pos,
            observation.prev_gate_left,
            observation.prev_gate_right,
            observation.curr_gate_left,
            observation.curr_gate_right,
        )
    if fraction is None:
        return None
    return observation.prev_timestamp_s + fraction * (
        observation.curr_timestamp_s - observation.prev_timestamp_s
    )


def build_gate_crossing_events(
    observations: list[GateCrossingObservation],
    config: TimingConfig,
) -> list[GateCrossingEvent]:
    """Erzeugt geordnete Torereignisse mit Start und Abschnittszeiten."""
    converted: list[tuple[GateCrossingObservation, float]] = []
    for observation in observations:
        timestamp = observation_to_timestamp(observation)
        if timestamp is not None:
            converted.append((observation, timestamp))
    converted.sort(key=lambda item: item[1])

    if config.start_mode == StartMode.MANUAL_FRAME:
        assert config.manual_start_frame is not None
        start_timestamp = config.manual_start_frame / config.video_fps
    else:
        start_timestamp = next(
            (
                timestamp
                for observation, timestamp in converted
                if observation.gate_id == config.start_gate_id
            ),
            None,
        )
        if start_timestamp is None:
            raise ValueError(f"Starttor {config.start_gate_id} wurde nicht durchfahren")

    events: list[GateCrossingEvent] = []
    previous_timestamp: float | None = None
    for observation, timestamp in converted:
        time_since_start = timestamp - start_timestamp
        if time_since_start < 0:
            continue
        split_time = None if previous_timestamp is None else timestamp - previous_timestamp
        events.append(
            GateCrossingEvent(
                gate_id=observation.gate_id,
                frame_number=observation.curr_frame_number,
                video_timestamp_s=timestamp,
                time_since_start_s=time_since_start,
                split_time_s=split_time,
                confidence=observation.confidence,
                uncertain=observation.uncertain,
                uncertain_reason=observation.uncertain_reason,
            )
        )
        previous_timestamp = timestamp
    return events


def write_gate_events_csv(events: list[GateCrossingEvent], path: str | Path) -> Path:
    """Schreibt Torereignisse in eine CSV Datei unter outputs oder einem Zielpfad."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(GateCrossingEvent.__dataclass_fields__)
    with target.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(event) for event in events)
    return target
