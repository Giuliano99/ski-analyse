import pytest

from src.tracking.athlete_tracker import (
    AthleteSelection,
    AthleteTrackObservation,
    load_selection,
    save_selection,
    select_nearest_person,
)
from src.detection.person_detector import PersonDetection


def test_athlete_selection_round_trip_and_reference_point(tmp_path):
    selection = AthleteSelection(
        video_name="run3.mp4",
        frame_number=12,
        timestamp_s=0.4,
        x=10,
        y=20,
        width=30,
        height=40,
    )

    target = save_selection(selection, tmp_path / "run3.json")

    assert load_selection(target) == selection
    assert selection.athlete_point.x == pytest.approx(25)
    assert selection.athlete_point.y == pytest.approx(60)


def test_athlete_selection_rejects_empty_box():
    with pytest.raises(ValueError, match="positive Groesse"):
        AthleteSelection("run.mp4", 0, 0, 0, 0, 0, 10)


def test_track_observation_uses_bottom_center_as_athlete_point():
    observation = AthleteTrackObservation.from_bbox(3, 0.1, (10, 20, 30, 40))

    assert observation.athlete_point is not None
    assert observation.athlete_point.x == pytest.approx(25)
    assert observation.athlete_point.y == pytest.approx(60)


def test_untracked_observation_has_no_athlete_point():
    observation = AthleteTrackObservation(4, 0.2, False)

    assert observation.athlete_point is None


def test_select_nearest_person_uses_footpoint_distance():
    detections = [
        PersonDetection((80, 80, 20, 40), 0.9),
        PersonDetection((12, 8, 20, 40), 0.3),
    ]

    selected = select_nearest_person(detections, (10, 10, 20, 40), max_distance_px=20)

    assert selected == detections[1]


def test_select_nearest_person_rejects_distant_candidates():
    detections = [PersonDetection((200, 200, 20, 40), 0.9)]

    assert select_nearest_person(detections, (10, 10, 20, 40), max_distance_px=30) is None

