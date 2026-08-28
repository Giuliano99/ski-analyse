from src.annotation.manual_annotations import (
    FrameAnnotation,
    ManualAnnotationDocument,
    ManualGateCrossing,
    load_document,
    save_document,
    to_observations,
)
from src.timing.geometry import Point
from src.timing.lap_time import StartMode, TimingConfig, build_gate_crossing_events


def frame(number: int, athlete_y: float) -> FrameAnnotation:
    return FrameAnnotation(
        frame_number=number,
        timestamp_s=number / 30,
        athlete_pos=Point(5, athlete_y),
        gate_left=Point(0, 0),
        gate_right=Point(10, 0),
    )


def test_annotation_round_trip_and_timing(tmp_path):
    document = ManualAnnotationDocument(
        video_name="run3.mp4",
        width=1920,
        height=1080,
        fps=29.871,
        target_athlete="Athlet in Rot",
    )
    document.add_crossing(
        ManualGateCrossing(1, frame(10, -1), frame(11, 1))
    )
    target = save_document(document, tmp_path / "run3.json")

    loaded = load_document(target)
    observations = to_observations(loaded)
    events = build_gate_crossing_events(
        observations,
        TimingConfig(StartMode.FIRST_GATE, video_fps=loaded.fps),
    )

    assert loaded == document
    assert len(events) == 1
    assert events[0].time_since_start_s == 0


def test_duplicate_gate_id_is_rejected():
    document = ManualAnnotationDocument("run.mp4", 100, 100, 30, "Athlet")
    crossing = ManualGateCrossing(1, frame(1, -1), frame(2, 1))
    document.add_crossing(crossing)

    try:
        document.add_crossing(crossing)
    except ValueError as error:
        assert "bereits vorhanden" in str(error)
    else:
        raise AssertionError("Doppelte gate_id wurde akzeptiert")


def test_manual_crossing_without_side_change_uses_uncertain_midpoint():
    document = ManualAnnotationDocument("run.mp4", 100, 100, 30, "Athlet")
    document.add_crossing(
        ManualGateCrossing(1, frame(10, -2), frame(11, -1))
    )

    events = build_gate_crossing_events(
        to_observations(document),
        TimingConfig(StartMode.FIRST_GATE, video_fps=document.fps),
    )

    assert len(events) == 1
    assert events[0].video_timestamp_s == (10.5 / 30)
    assert events[0].uncertain is True
    assert "ohne geometrischen Seitenwechsel" in events[0].uncertain_reason
