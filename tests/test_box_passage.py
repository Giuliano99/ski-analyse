import pytest

from src.timing.box_passage import find_box_passage_candidates
from src.timing.geometry import BoundingBox
from src.tracking.athlete_tracker import AthleteTrackObservation
from src.tracking.gate_tracker import GateTrack, GateTrackObservation


def athlete(frame: int, y: float, *, uncertain: bool = False):
    return AthleteTrackObservation.from_bbox(
        frame,
        frame / 10,
        (2, y, 6, 2),
        uncertain=uncertain,
    )


def gate(frame: int, *, uncertain: bool = False):
    return GateTrackObservation(
        frame,
        frame / 10,
        BoundingBox(0, -1, 10, 0),
        0.8,
        uncertain,
    )


def test_candidate_uses_interpolated_frame_and_timestamp():
    candidates = find_box_passage_candidates(
        [athlete(10, -4), athlete(12, 0)],
        [GateTrack(7, [gate(10), gate(12)])],
    )

    assert len(candidates) == 1
    assert candidates[0].gate_track_id == 7
    assert candidates[0].frame_number == 11
    assert candidates[0].timestamp_s == pytest.approx(1.1)
    assert candidates[0].uncertain is False


def test_candidate_propagates_athlete_and_gate_uncertainty():
    candidates = find_box_passage_candidates(
        [athlete(10, -4, uncertain=True), athlete(12, 0)],
        [GateTrack(7, [gate(10), gate(12, uncertain=True)])],
    )

    assert candidates[0].uncertain is True
    assert "Athletenspur" in candidates[0].uncertain_reason
    assert "Tortrack" in candidates[0].uncertain_reason


def test_untracked_athlete_interval_is_skipped():
    missing = AthleteTrackObservation(12, 1.2, False)
    candidates = find_box_passage_candidates(
        [athlete(10, -4), missing],
        [GateTrack(7, [gate(10), gate(12)])],
    )

    assert candidates == []
