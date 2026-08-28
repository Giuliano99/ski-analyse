"""Deterministische Zuordnung von Torboxen ueber aufeinanderfolgende Frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from src.detection.gate_detector import GateDetection
from src.timing.geometry import BoundingBox


@dataclass(frozen=True)
class GateDetectionFrame:
    frame_number: int
    timestamp_s: float
    detections: list[GateDetection]


@dataclass(frozen=True)
class GateTrackObservation:
    frame_number: int
    timestamp_s: float
    box: BoundingBox
    confidence: float
    uncertain: bool


@dataclass
class GateTrack:
    track_id: int
    observations: list[GateTrackObservation] = field(default_factory=list)


@dataclass(frozen=True)
class GateTrackerConfig:
    max_gap_frames: int = 2
    min_iou: float = 0.05
    max_center_distance_ratio: float = 1.5

    def __post_init__(self) -> None:
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames darf nicht negativ sein")
        if not 0.0 <= self.min_iou <= 1.0:
            raise ValueError("min_iou muss zwischen 0 und 1 liegen")
        if self.max_center_distance_ratio <= 0:
            raise ValueError("max_center_distance_ratio muss positiv sein")


def track_gate_detections(
    frames: list[GateDetectionFrame],
    config: GateTrackerConfig = GateTrackerConfig(),
) -> list[GateTrack]:
    """Verknuepft Detektionen per Bewegungsprognose, Distanz und IoU."""

    ordered = sorted(frames, key=lambda item: item.frame_number)
    if len({item.frame_number for item in ordered}) != len(ordered):
        raise ValueError("Detektionsframes duerfen nicht doppelt vorkommen")

    tracks: list[GateTrack] = []
    next_track_id = 1
    for frame in ordered:
        boxes = [_detection_box(item) for item in frame.detections]
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            last = track.observations[-1]
            gap = frame.frame_number - last.frame_number
            if gap <= 0 or gap > config.max_gap_frames + 1:
                continue
            predicted = _predicted_box(track, frame.frame_number)
            for detection_index, box in enumerate(boxes):
                iou = _iou(predicted, box)
                distance_ratio = _center_distance_ratio(predicted, box)
                if iou < config.min_iou and distance_ratio > config.max_center_distance_ratio:
                    continue
                candidates.append((distance_ratio - iou, track_index, detection_index))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_index, detection_index in sorted(candidates):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            track = tracks[track_index]
            gap = frame.frame_number - track.observations[-1].frame_number
            detection = frame.detections[detection_index]
            track.observations.append(
                GateTrackObservation(
                    frame.frame_number,
                    frame.timestamp_s,
                    boxes[detection_index],
                    detection.confidence,
                    uncertain=gap > 1,
                )
            )
            used_tracks.add(track_index)
            used_detections.add(detection_index)

        for detection_index, detection in enumerate(frame.detections):
            if detection_index in used_detections:
                continue
            tracks.append(
                GateTrack(
                    next_track_id,
                    [
                        GateTrackObservation(
                            frame.frame_number,
                            frame.timestamp_s,
                            boxes[detection_index],
                            detection.confidence,
                            uncertain=False,
                        )
                    ],
                )
            )
            next_track_id += 1
    return tracks


def _detection_box(detection: GateDetection) -> BoundingBox:
    return BoundingBox.from_center_xywh(
        detection.x, detection.y, detection.width, detection.height
    )


def _predicted_box(track: GateTrack, frame_number: int) -> BoundingBox:
    last = track.observations[-1]
    if len(track.observations) < 2:
        return last.box
    previous = track.observations[-2]
    frame_delta = last.frame_number - previous.frame_number
    if frame_delta <= 0:
        return last.box
    prediction_delta = frame_number - last.frame_number
    dx = (last.box.center.x - previous.box.center.x) / frame_delta * prediction_delta
    dy = (last.box.center.y - previous.box.center.y) / frame_delta * prediction_delta
    return BoundingBox(
        last.box.left + dx,
        last.box.top + dy,
        last.box.right + dx,
        last.box.bottom + dy,
    )


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0.0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = width * height
    if intersection == 0:
        return 0.0
    first_area = (first.right - first.left) * (first.bottom - first.top)
    second_area = (second.right - second.left) * (second.bottom - second.top)
    return intersection / (first_area + second_area - intersection)


def _center_distance_ratio(first: BoundingBox, second: BoundingBox) -> float:
    diagonal = hypot(first.right - first.left, first.bottom - first.top)
    return hypot(
        first.center.x - second.center.x,
        first.center.y - second.center.y,
    ) / max(diagonal, 1.0)
