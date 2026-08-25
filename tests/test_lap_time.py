import csv

import pytest

from src.timing.geometry import Point
from src.timing.lap_time import (
    GateCrossingObservation,
    StartMode,
    TimingConfig,
    build_gate_crossing_events,
    write_gate_events_csv,
)


def observation(gate_id: int, start_s: float, end_s: float) -> GateCrossingObservation:
    return GateCrossingObservation(
        gate_id=gate_id,
        prev_frame_number=round(start_s * 10),
        curr_frame_number=round(end_s * 10),
        prev_timestamp_s=start_s,
        curr_timestamp_s=end_s,
        prev_athlete_pos=Point(5, -1),
        curr_athlete_pos=Point(5, 1),
        prev_gate_left=Point(0, 0),
        prev_gate_right=Point(10, 0),
        curr_gate_left=Point(0, 0),
        curr_gate_right=Point(10, 0),
    )


def test_first_gate_is_time_zero_and_splits_are_interpolated():
    events = build_gate_crossing_events(
        [observation(1, 1.0, 1.1), observation(2, 3.0, 3.1)],
        TimingConfig(StartMode.FIRST_GATE, video_fps=10),
    )

    assert events[0].time_since_start_s == pytest.approx(0)
    assert events[0].split_time_s is None
    assert events[1].time_since_start_s == pytest.approx(2)
    assert events[1].split_time_s == pytest.approx(2)


def test_manual_frame_is_time_zero():
    events = build_gate_crossing_events(
        [observation(1, 1.0, 1.1)],
        TimingConfig(StartMode.MANUAL_FRAME, video_fps=10, manual_start_frame=5),
    )

    assert events[0].time_since_start_s == pytest.approx(0.55)


def test_missing_first_gate_is_explicit_error():
    with pytest.raises(ValueError, match="Starttor 1"):
        build_gate_crossing_events(
            [observation(2, 1.0, 1.1)],
            TimingConfig(StartMode.FIRST_GATE, video_fps=10),
        )


def test_uncertainty_is_preserved_in_csv(tmp_path):
    item = observation(1, 1.0, 1.1)
    item = GateCrossingObservation(
        **{**item.__dict__, "uncertain": True, "uncertain_reason": "Tor verdeckt"}
    )
    events = build_gate_crossing_events(
        [item], TimingConfig(StartMode.FIRST_GATE, video_fps=10)
    )
    target = write_gate_events_csv(events, tmp_path / "events.csv")

    with target.open(encoding="utf-8", newline="") as csv_file:
        row = next(csv.DictReader(csv_file))
    assert row["uncertain"] == "True"
    assert row["uncertain_reason"] == "Tor verdeckt"
