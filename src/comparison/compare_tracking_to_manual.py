"""Automatische Athletenspur gegen manuelle Torzeiten vergleichen."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from src.annotation.manual_annotations import (
    ManualAnnotationDocument,
    ManualGateCrossing,
    load_document,
    to_observations,
)
from src.timing.geometry import Point, moving_gate_crossing_fraction
from src.timing.lap_time import observation_to_timestamp
from src.tracking.athlete_tracker import (
    AthleteTrackDocument,
    AthleteTrackObservation,
    load_track,
)


@dataclass
class GateTimeComparison:
    gate_id: int
    status: str
    reference_frame_number: int
    tracked_frame_number: int | None
    reference_timestamp_s: float
    tracked_timestamp_s: float | None
    timestamp_error_ms: float | None
    reference_time_since_start_s: float
    tracked_time_since_start_s: float | None
    time_since_start_error_ms: float | None
    reference_split_s: float | None
    tracked_split_s: float | None
    split_error_ms: float | None
    confidence: float | None
    reference_uncertain: bool
    reference_uncertain_reason: str | None
    uncertain: bool
    uncertain_reason: str | None


@dataclass(frozen=True)
class _TrackedCrossing:
    frame_number: int
    timestamp_s: float
    confidence: float | None
    uncertain: bool
    uncertain_reason: str | None


def compare_track_to_manual(
    document: ManualAnnotationDocument,
    track: AthleteTrackDocument,
    *,
    window_frames: int = 10,
) -> list[GateTimeComparison]:
    if window_frames < 0:
        raise ValueError("window_frames darf nicht negativ sein")
    if document.video_name != track.video_name:
        raise ValueError("Annotation und Athletenspur gehoeren zu verschiedenen Videos")

    reference_observations = to_observations(document)
    reference_by_gate = {item.gate_id: item for item in reference_observations}
    reference_timestamps = {
        item.gate_id: observation_to_timestamp(item) for item in reference_observations
    }
    frames = {item.frame_number: item for item in track.frames}
    tracked_crossings: dict[int, _TrackedCrossing | None] = {}
    for crossing in document.crossings:
        reference_timestamp = reference_timestamps[crossing.gate_id]
        assert reference_timestamp is not None
        tracked_crossings[crossing.gate_id] = _find_tracked_crossing(
            crossing,
            frames,
            reference_timestamp=reference_timestamp,
            window_frames=window_frames,
        )

    ordered = sorted(document.crossings, key=lambda item: reference_timestamps[item.gate_id])
    reference_start = reference_timestamps[ordered[0].gate_id]
    assert reference_start is not None
    first_tracked = tracked_crossings[ordered[0].gate_id]
    tracked_start = first_tracked.timestamp_s if first_tracked is not None else None

    comparisons: list[GateTimeComparison] = []
    previous_reference: float | None = None
    previous_tracked: float | None = None
    for crossing in ordered:
        reference_timestamp = reference_timestamps[crossing.gate_id]
        assert reference_timestamp is not None
        tracked = tracked_crossings[crossing.gate_id]
        reference_observation = reference_by_gate[crossing.gate_id]
        reference_elapsed = reference_timestamp - reference_start
        reference_split = (
            None if previous_reference is None else reference_timestamp - previous_reference
        )
        tracked_elapsed = (
            None
            if tracked is None or tracked_start is None
            else tracked.timestamp_s - tracked_start
        )
        tracked_split = (
            None
            if tracked is None or previous_tracked is None
            else tracked.timestamp_s - previous_tracked
        )
        reasons = [
            reason
            for reason in (
                reference_observation.uncertain_reason,
                (
                    "Kein geometrischer Seitenwechsel im Suchfenster"
                    if tracked is None
                    else tracked.uncertain_reason
                ),
            )
            if reason
        ]
        comparisons.append(
            GateTimeComparison(
                gate_id=crossing.gate_id,
                status="matched" if tracked is not None else "no_crossing",
                reference_frame_number=crossing.current.frame_number,
                tracked_frame_number=tracked.frame_number if tracked else None,
                reference_timestamp_s=reference_timestamp,
                tracked_timestamp_s=tracked.timestamp_s if tracked else None,
                timestamp_error_ms=(
                    None
                    if tracked is None
                    else (tracked.timestamp_s - reference_timestamp) * 1000.0
                ),
                reference_time_since_start_s=reference_elapsed,
                tracked_time_since_start_s=tracked_elapsed,
                time_since_start_error_ms=(
                    None
                    if tracked_elapsed is None
                    else (tracked_elapsed - reference_elapsed) * 1000.0
                ),
                reference_split_s=reference_split,
                tracked_split_s=tracked_split,
                split_error_ms=(
                    None
                    if reference_split is None or tracked_split is None
                    else (tracked_split - reference_split) * 1000.0
                ),
                confidence=tracked.confidence if tracked else None,
                reference_uncertain=reference_observation.uncertain,
                reference_uncertain_reason=reference_observation.uncertain_reason,
                uncertain=(
                    reference_observation.uncertain
                    or tracked is None
                    or tracked.uncertain
                ),
                uncertain_reason="; ".join(reasons) or None,
            )
        )
        previous_reference = reference_timestamp
        previous_tracked = tracked.timestamp_s if tracked is not None else None
    return comparisons


def _find_tracked_crossing(
    crossing: ManualGateCrossing,
    frames: dict[int, AthleteTrackObservation],
    *,
    reference_timestamp: float,
    window_frames: int,
) -> _TrackedCrossing | None:
    start = max(0, crossing.previous.frame_number - window_frames)
    end = crossing.current.frame_number + window_frames
    candidates: list[_TrackedCrossing] = []
    for frame_number in range(start, end):
        previous = frames.get(frame_number)
        current = frames.get(frame_number + 1)
        if previous is None or current is None:
            continue
        previous_point = previous.athlete_point
        current_point = current.athlete_point
        if previous_point is None or current_point is None:
            continue
        previous_left, previous_right = _gate_at_frame(crossing, frame_number)
        current_left, current_right = _gate_at_frame(crossing, frame_number + 1)
        fraction = moving_gate_crossing_fraction(
            previous_point,
            current_point,
            previous_left,
            previous_right,
            current_left,
            current_right,
        )
        if fraction is None:
            continue
        timestamp = previous.timestamp_s + fraction * (
            current.timestamp_s - previous.timestamp_s
        )
        uncertain_frames = [item for item in (previous, current) if item.uncertain]
        reasons = [
            f"Athletenspur in Frame {item.frame_number} ist unsicher ({item.source})"
            for item in uncertain_frames
        ]
        confidences = [
            item.confidence
            for item in (previous, current)
            if item.confidence is not None
        ]
        candidates.append(
            _TrackedCrossing(
                frame_number=current.frame_number,
                timestamp_s=timestamp,
                confidence=min(confidences) if confidences else None,
                uncertain=bool(uncertain_frames),
                uncertain_reason="; ".join(reasons) or None,
            )
        )
    return min(
        candidates,
        key=lambda item: abs(item.timestamp_s - reference_timestamp),
        default=None,
    )


def _gate_at_frame(
    crossing: ManualGateCrossing,
    frame_number: int,
) -> tuple[Point, Point]:
    frame_delta = crossing.current.frame_number - crossing.previous.frame_number
    fraction = (frame_number - crossing.previous.frame_number) / frame_delta

    def extrapolate(previous: Point, current: Point) -> Point:
        return Point(
            previous.x + fraction * (current.x - previous.x),
            previous.y + fraction * (current.y - previous.y),
        )

    return (
        extrapolate(crossing.previous.gate_left, crossing.current.gate_left),
        extrapolate(crossing.previous.gate_right, crossing.current.gate_right),
    )


def write_comparison_csv(
    comparisons: list[GateTimeComparison],
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(GateTimeComparison.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(item) for item in comparisons)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("--window-frames", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    document = load_document(args.annotation)
    comparisons = compare_track_to_manual(
        document,
        load_track(args.track),
        window_frames=args.window_frames,
    )
    output = args.output or (
        Path("outputs/timing_comparison") / f"{Path(document.video_name).stem}.csv"
    )
    write_comparison_csv(comparisons, output)
    matched = [item for item in comparisons if item.timestamp_error_ms is not None]
    reliable = [item for item in matched if not item.uncertain]
    mean_absolute_error = (
        sum(abs(item.timestamp_error_ms) for item in matched) / len(matched)
        if matched
        else None
    )
    error_text = "n/a" if mean_absolute_error is None else f"{mean_absolute_error:.1f} ms"
    print(
        f"{len(matched)}/{len(comparisons)} Tore geometrisch zugeordnet: {output} "
        f"(Roh-MAE {error_text}, {len(reliable)} verlaesslich, "
        f"{sum(item.uncertain for item in comparisons)} unsicher)"
    )


if __name__ == "__main__":
    main()
