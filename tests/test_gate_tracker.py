from src.detection.gate_detector import GateDetection
from src.tracking.gate_tracker import (
    GateDetectionFrame,
    GateTrackerConfig,
    track_gate_detections,
)


def detection(x: float, y: float = 50) -> GateDetection:
    return GateDetection(x, y, 20, 30, 0.8, "Ski-Gates")


def test_gate_keeps_track_id_during_steady_camera_motion():
    tracks = track_gate_detections(
        [
            GateDetectionFrame(0, 0.0, [detection(10)]),
            GateDetectionFrame(1, 0.04, [detection(20)]),
            GateDetectionFrame(2, 0.08, [detection(30)]),
        ]
    )

    assert len(tracks) == 1
    assert [item.frame_number for item in tracks[0].observations] == [0, 1, 2]


def test_two_gates_are_not_merged():
    tracks = track_gate_detections(
        [
            GateDetectionFrame(0, 0.0, [detection(10), detection(200)]),
            GateDetectionFrame(1, 0.04, [detection(12), detection(198)]),
        ]
    )

    assert len(tracks) == 2
    assert all(len(track.observations) == 2 for track in tracks)


def test_reassociation_after_missing_frame_is_uncertain():
    tracks = track_gate_detections(
        [
            GateDetectionFrame(0, 0.0, [detection(10)]),
            GateDetectionFrame(1, 0.04, []),
            GateDetectionFrame(2, 0.08, [detection(12)]),
        ],
        GateTrackerConfig(max_gap_frames=1),
    )

    assert len(tracks) == 1
    assert tracks[0].observations[-1].uncertain is True


def test_duplicate_frame_numbers_are_rejected():
    frames = [
        GateDetectionFrame(0, 0.0, []),
        GateDetectionFrame(0, 0.1, []),
    ]

    try:
        track_gate_detections(frames)
    except ValueError as error:
        assert "doppelt" in str(error)
    else:
        raise AssertionError("Doppelte Frames wurden akzeptiert")
