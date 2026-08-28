"""Datenmodell fuer die manuelle Initialisierung des Athletentrackings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from src.detection.person_detector import LocalPersonDetector, PersonDetection
from src.timing.geometry import Point


@dataclass(frozen=True)
class AthleteSelection:
    video_name: str
    frame_number: int
    timestamp_s: float
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.frame_number < 0:
            raise ValueError("frame_number darf nicht negativ sein")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Athletenbox muss eine positive Groesse haben")

    @property
    def athlete_point(self) -> Point:
        """Fusspunktmitte als Bezugspunkt fuer die Durchfahrtsregel."""

        return Point(self.x + self.width / 2, self.y + self.height)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height


def save_selection(selection: AthleteSelection, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as output_file:
        json.dump(asdict(selection), output_file, indent=2)
    return target


def load_selection(path: str | Path) -> AthleteSelection:
    with Path(path).open(encoding="utf-8") as input_file:
        return AthleteSelection(**json.load(input_file))


@dataclass(frozen=True)
class AthleteTrackObservation:
    frame_number: int
    timestamp_s: float
    tracked: bool
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    source: str = "unknown"
    confidence: float | None = None
    uncertain: bool = True

    @classmethod
    def from_bbox(
        cls,
        frame_number: int,
        timestamp_s: float,
        bbox: tuple[float, float, float, float],
        *,
        source: str = "tracker",
        confidence: float | None = None,
        uncertain: bool = True,
    ) -> "AthleteTrackObservation":
        x, y, width, height = bbox
        return cls(
            frame_number,
            timestamp_s,
            True,
            x,
            y,
            width,
            height,
            source,
            confidence,
            uncertain,
        )

    @property
    def athlete_point(self) -> Point | None:
        if not self.tracked or None in (self.x, self.y, self.width, self.height):
            return None
        assert self.x is not None
        assert self.y is not None
        assert self.width is not None
        assert self.height is not None
        return Point(self.x + self.width / 2, self.y + self.height)


@dataclass(frozen=True)
class AthleteTrackDocument:
    video_name: str
    selection: AthleteSelection
    frames: list[AthleteTrackObservation]


def load_track(path: str | Path) -> AthleteTrackDocument:
    with Path(path).open(encoding="utf-8") as input_file:
        data = json.load(input_file)
    frames = [AthleteTrackObservation(**item) for item in data["frames"]]
    frame_numbers = [item.frame_number for item in frames]
    if len(frame_numbers) != len(set(frame_numbers)):
        raise ValueError("Athletenspur enthaelt doppelte Framenummern")
    return AthleteTrackDocument(
        video_name=str(data["video_name"]),
        selection=AthleteSelection(**data["selection"]),
        frames=frames,
    )


def save_track(
    video_path: Path,
    selection: AthleteSelection,
    observations: list[AthleteTrackObservation],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "video_name": video_path.name,
        "selection": asdict(selection),
        "frames": [asdict(observation) for observation in observations],
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2)
    return output_path


def track_athlete(
    video_path: Path,
    selection: AthleteSelection,
    output_path: Path,
    preview_dir: Path,
    *,
    preview_every: int = 30,
    tracker_type: str = "mil",
) -> Path:
    if selection.video_name != video_path.name:
        raise ValueError("Athletenauswahl gehoert zu einem anderen Video")
    if preview_every <= 0:
        raise ValueError("preview_every muss positiv sein")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video kann nicht geoeffnet werden: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, selection.frame_number)
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise ValueError(f"Startframe {selection.frame_number} kann nicht gelesen werden")

    tracker = _create_tracker(tracker_type)
    initial_bbox = tuple(round(value) for value in selection.bbox)
    tracker.init(frame, initial_bbox)
    preview_dir.mkdir(parents=True, exist_ok=True)
    observations: list[AthleteTrackObservation] = []
    frame_number = selection.frame_number

    try:
        while ok:
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_s = timestamp_ms / 1000.0
            if frame_number == selection.frame_number:
                tracked, bbox = True, selection.bbox
            else:
                tracked, bbox = tracker.update(frame)

            if tracked:
                observation = AthleteTrackObservation.from_bbox(
                    frame_number,
                    timestamp_s,
                    tuple(float(value) for value in bbox),
                    source="manual" if frame_number == selection.frame_number else tracker_type,
                    uncertain=frame_number != selection.frame_number,
                )
            else:
                observation = AthleteTrackObservation(frame_number, timestamp_s, False)
            observations.append(observation)

            if (frame_number - selection.frame_number) % preview_every == 0:
                preview = _draw_track_preview(frame, observation)
                cv2.imwrite(str(preview_dir / f"frame_{frame_number:06d}.jpg"), preview)

            ok, frame = capture.read()
            frame_number += 1
    finally:
        capture.release()

    return save_track(video_path, selection, observations, output_path)


def bbox_footpoint(bbox: tuple[float, float, float, float]) -> Point:
    x, y, width, height = bbox
    return Point(x + width / 2, y + height)


def select_nearest_person(
    detections: list[PersonDetection],
    reference_bbox: tuple[float, float, float, float],
    *,
    max_distance_px: float,
) -> PersonDetection | None:
    """Waehlt nur eine Person nahe der erwarteten Position."""

    if max_distance_px <= 0:
        raise ValueError("max_distance_px muss positiv sein")
    reference = bbox_footpoint(reference_bbox)
    candidates = [
        detection
        for detection in detections
        if _point_distance(reference, bbox_footpoint(detection.bbox)) <= max_distance_px
    ]
    ranked = sorted(
        candidates,
        key=lambda detection: _point_distance(reference, bbox_footpoint(detection.bbox)),
    )
    if not ranked:
        return None
    return ranked[0]


def track_athlete_hybrid(
    video_path: Path,
    selection: AthleteSelection,
    output_path: Path,
    preview_dir: Path,
    *,
    model_path: str | Path = "yolo11n.pt",
    detector_confidence: float = 0.05,
    image_size: int = 1280,
    tracker_type: str = "csrt",
    confirmation_distance_px: float = 90.0,
    confirmation_grace_frames: int = 10,
    recovery_distance_px: float = 450.0,
    recovery_confidence: float = 0.25,
    preview_every: int = 30,
) -> Path:
    """Verfolgt primaer mit CSRT und nutzt YOLO nur zur Kontrolle/Erholung."""

    if selection.video_name != video_path.name:
        raise ValueError("Athletenauswahl gehoert zu einem anderen Video")
    if preview_every <= 0 or confirmation_grace_frames < 0:
        raise ValueError("preview_every muss positiv und die Karenz nicht negativ sein")

    detector = LocalPersonDetector(
        model_path,
        confidence=detector_confidence,
        image_size=image_size,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video kann nicht geoeffnet werden: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, selection.frame_number)
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise ValueError(f"Startframe {selection.frame_number} kann nicht gelesen werden")

    tracker = _new_tracker(frame, selection.bbox, tracker_type)
    reference_bbox = selection.bbox
    frames_since_confirmation = 0
    observations: list[AthleteTrackObservation] = []
    preview_dir.mkdir(parents=True, exist_ok=True)
    frame_number = selection.frame_number

    try:
        while ok:
            timestamp_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if frame_number == selection.frame_number:
                tracked, tracker_bbox = True, selection.bbox
            else:
                tracked, tracker_bbox = tracker.update(frame)
                tracker_bbox = tuple(float(value) for value in tracker_bbox)
            detections = detector.detect(frame)

            if tracked:
                reference_bbox = tracker_bbox
                confirmation = select_nearest_person(
                    detections,
                    tracker_bbox,
                    max_distance_px=confirmation_distance_px,
                )
                if confirmation is not None:
                    frames_since_confirmation = 0
                    source = f"{tracker_type}+yolo"
                    confidence = confirmation.confidence
                else:
                    frames_since_confirmation += 1
                    source = tracker_type
                    confidence = None
                observation = AthleteTrackObservation.from_bbox(
                    frame_number,
                    timestamp_s,
                    tracker_bbox,
                    source="manual" if frame_number == selection.frame_number else source,
                    confidence=confidence,
                    uncertain=(
                        False
                        if frame_number == selection.frame_number
                        else frames_since_confirmation > confirmation_grace_frames
                    ),
                )
            else:
                recovery_candidates = [
                    detection
                    for detection in detections
                    if detection.confidence >= recovery_confidence
                ]
                recovery = select_nearest_person(
                    recovery_candidates,
                    reference_bbox,
                    max_distance_px=recovery_distance_px,
                )
                if recovery is not None:
                    reference_bbox = recovery.bbox
                    tracker = _new_tracker(frame, reference_bbox, tracker_type)
                    frames_since_confirmation = 0
                    observation = AthleteTrackObservation.from_bbox(
                        frame_number,
                        timestamp_s,
                        reference_bbox,
                        source="yolo_recovery",
                        confidence=recovery.confidence,
                        uncertain=recovery.confidence < 0.5,
                    )
                else:
                    frames_since_confirmation += 1
                    observation = AthleteTrackObservation(
                        frame_number,
                        timestamp_s,
                        False,
                        source="missing",
                        uncertain=True,
                    )
            observations.append(observation)

            if (frame_number - selection.frame_number) % preview_every == 0:
                preview = _draw_track_preview(frame, observation)
                cv2.imwrite(str(preview_dir / f"frame_{frame_number:06d}.jpg"), preview)

            ok, frame = capture.read()
            frame_number += 1
    finally:
        capture.release()

    return save_track(video_path, selection, observations, output_path)


def _create_tracker(tracker_type: str):
    if tracker_type == "mil":
        return cv2.TrackerMIL_create()
    if tracker_type == "csrt":
        if not hasattr(cv2, "TrackerCSRT_create"):
            raise RuntimeError("CSRT benoetigt opencv-contrib-python")
        return cv2.TrackerCSRT_create()
    raise ValueError(f"Unbekannter Tracker: {tracker_type}")


def _new_tracker(
    frame,
    bbox: tuple[float, float, float, float],
    tracker_type: str,
):
    tracker = _create_tracker(tracker_type)
    tracker.init(frame, tuple(round(value) for value in bbox))
    return tracker


def _point_distance(first: Point, second: Point) -> float:
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5


def _draw_track_preview(frame, observation: AthleteTrackObservation):
    preview = frame.copy()
    if observation.tracked:
        assert observation.x is not None
        assert observation.y is not None
        assert observation.width is not None
        assert observation.height is not None
        x1, y1 = round(observation.x), round(observation.y)
        x2 = round(observation.x + observation.width)
        y2 = round(observation.y + observation.height)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
        point = observation.athlete_point
        assert point is not None
        cv2.circle(preview, (round(point.x), round(point.y)), 6, (0, 0, 255), -1)
    cv2.putText(
        preview,
        (
            f"Frame {observation.frame_number} tracked={observation.tracked} "
            f"source={observation.source} uncertain={observation.uncertain}"
        ),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview
