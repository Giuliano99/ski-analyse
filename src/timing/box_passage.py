"""Boxbasierte Durchfahrtskandidaten aus Athleten- und Torspuren."""

from __future__ import annotations

from dataclasses import dataclass

from src.timing.geometry import BoundingBox, moving_box_gate_crossing
from src.tracking.athlete_tracker import AthleteTrackObservation
from src.tracking.gate_tracker import GateTrack


@dataclass(frozen=True)
class BoxPassageCandidate:
    gate_track_id: int
    frame_number: int
    timestamp_s: float
    crossing_fraction: float
    supporting_points: int
    confidence: float
    uncertain: bool
    uncertain_reason: str | None = None


def find_box_passage_candidates(
    athlete_frames: list[AthleteTrackObservation],
    gate_tracks: list[GateTrack],
) -> list[BoxPassageCandidate]:
    """Findet geometrische Schnitte, ohne Track-IDs als Tor-IDs auszugeben."""

    athlete_by_frame = {item.frame_number: item for item in athlete_frames}
    if len(athlete_by_frame) != len(athlete_frames):
        raise ValueError("Athletenspur enthaelt doppelte Framenummern")

    candidates: list[BoxPassageCandidate] = []
    for track in gate_tracks:
        observations = sorted(track.observations, key=lambda item: item.frame_number)
        for previous_gate, current_gate in zip(observations, observations[1:]):
            previous_athlete = athlete_by_frame.get(previous_gate.frame_number)
            current_athlete = athlete_by_frame.get(current_gate.frame_number)
            if not _has_box(previous_athlete) or not _has_box(current_athlete):
                continue
            assert previous_athlete is not None
            assert current_athlete is not None
            crossing = moving_box_gate_crossing(
                _athlete_box(previous_athlete),
                _athlete_box(current_athlete),
                previous_gate.box,
                current_gate.box,
            )
            if crossing is None:
                continue

            fraction = crossing.fraction
            frame = round(
                previous_gate.frame_number
                + fraction * (current_gate.frame_number - previous_gate.frame_number)
            )
            timestamp = previous_gate.timestamp_s + fraction * (
                current_gate.timestamp_s - previous_gate.timestamp_s
            )
            reasons: list[str] = []
            if crossing.uncertain:
                reasons.append("nur ein Box-Stuetzpunkt")
            if previous_gate.uncertain or current_gate.uncertain:
                reasons.append("Tortrack mit Erkennungsluecke")
            if previous_athlete.uncertain or current_athlete.uncertain:
                reasons.append("Athletenspur unsicher")
            candidates.append(
                BoxPassageCandidate(
                    gate_track_id=track.track_id,
                    frame_number=frame,
                    timestamp_s=timestamp,
                    crossing_fraction=fraction,
                    supporting_points=crossing.supporting_points,
                    confidence=min(previous_gate.confidence, current_gate.confidence),
                    uncertain=bool(reasons),
                    uncertain_reason="; ".join(reasons) or None,
                )
            )
    return sorted(candidates, key=lambda item: (item.frame_number, item.gate_track_id))


def _has_box(observation: AthleteTrackObservation | None) -> bool:
    return observation is not None and observation.tracked and None not in (
        observation.x,
        observation.y,
        observation.width,
        observation.height,
    )


def _athlete_box(observation: AthleteTrackObservation) -> BoundingBox:
    assert observation.x is not None
    assert observation.y is not None
    assert observation.width is not None
    assert observation.height is not None
    return BoundingBox.from_top_left_xywh(
        observation.x, observation.y, observation.width, observation.height
    )
