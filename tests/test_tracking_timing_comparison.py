import pytest

from src.annotation.manual_annotations import (
    FrameAnnotation,
    ManualAnnotationDocument,
    ManualGateCrossing,
)
from src.comparison.compare_tracking_to_manual import compare_track_to_manual
from src.timing.geometry import Point
from src.tracking.athlete_tracker import (
    AthleteSelection,
    AthleteTrackDocument,
    AthleteTrackObservation,
)


def _frame(number: int, athlete_y: float, gate_y: float = 0) -> FrameAnnotation:
    return FrameAnnotation(
        frame_number=number,
        timestamp_s=number / 10,
        athlete_pos=Point(5, athlete_y),
        gate_left=Point(0, gate_y),
        gate_right=Point(10, gate_y),
    )


def _track(points: list[tuple[int, float, bool]]) -> AthleteTrackDocument:
    return AthleteTrackDocument(
        video_name="run.mp4",
        selection=AthleteSelection("run.mp4", 0, 0, 4, -2, 2, 1),
        frames=[
            AthleteTrackObservation.from_bbox(
                number,
                number / 10,
                (4, y - 1, 2, 1),
                source="csrt",
                uncertain=uncertain,
            )
            for number, y, uncertain in points
        ],
    )


def test_comparison_finds_crossing_near_manual_frames():
    document = ManualAnnotationDocument("run.mp4", 100, 100, 10, "rot")
    document.add_crossing(ManualGateCrossing(1, _frame(10, -1), _frame(11, 1)))
    track = _track([(8, -2, False), (9, -1, False), (10, 1, False), (11, 2, False)])

    result = compare_track_to_manual(document, track, window_frames=2)

    assert result[0].status == "matched"
    assert result[0].tracked_frame_number == 10
    assert result[0].tracked_timestamp_s == pytest.approx(0.95)
    assert result[0].timestamp_error_ms == pytest.approx(-100)
    assert result[0].uncertain is False


def test_comparison_keeps_missing_crossing_visible():
    document = ManualAnnotationDocument("run.mp4", 100, 100, 10, "rot")
    document.add_crossing(ManualGateCrossing(1, _frame(10, -1), _frame(11, 1)))
    track = _track([(9, -2, False), (10, -1, False), (11, -0.5, False)])

    result = compare_track_to_manual(document, track, window_frames=1)

    assert result[0].status == "no_crossing"
    assert result[0].tracked_timestamp_s is None
    assert result[0].uncertain is True


def test_comparison_propagates_uncertain_tracking_frame():
    document = ManualAnnotationDocument("run.mp4", 100, 100, 10, "rot")
    document.add_crossing(ManualGateCrossing(1, _frame(10, -1), _frame(11, 1)))
    track = _track([(10, -1, True), (11, 1, False)])

    result = compare_track_to_manual(document, track)

    assert result[0].status == "matched"
    assert result[0].uncertain is True
    assert "Frame 10" in result[0].uncertain_reason


def test_comparison_propagates_uncertain_manual_reference():
    document = ManualAnnotationDocument("run.mp4", 100, 100, 10, "rot")
    document.add_crossing(ManualGateCrossing(1, _frame(10, -2), _frame(11, -1)))
    track = _track([(10, -1, False), (11, 1, False)])

    result = compare_track_to_manual(document, track)

    assert result[0].reference_uncertain is True
    assert result[0].uncertain is True
    assert "ohne geometrischen Seitenwechsel" in result[0].reference_uncertain_reason
