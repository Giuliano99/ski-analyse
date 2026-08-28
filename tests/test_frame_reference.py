import pytest

from src.comparison.frame_reference import (
    FrameReferenceDocument,
    GateFrameReference,
    evaluate_predicted_frames,
    load_frame_references,
)


def test_load_and_evaluate_frame_references(tmp_path):
    path = tmp_path / "reference.yaml"
    path.write_text(
        "video_name: run.mp4\n"
        "tolerance_frames: 2\n"
        "crossings:\n"
        "  - gate_id: 1\n"
        "    frame_number: 70\n"
        "  - gate_id: 2\n"
        "    frame_number: 124\n",
        encoding="utf-8",
    )

    references = load_frame_references(path)
    result = evaluate_predicted_frames(references, {1: 72, 2: 121})

    assert result[0].frame_error == 2
    assert result[0].within_tolerance is True
    assert result[1].frame_error == -3
    assert result[1].within_tolerance is False


def test_missing_prediction_is_not_within_tolerance():
    references = FrameReferenceDocument(
        "run.mp4",
        2,
        [GateFrameReference(1, 70)],
    )

    result = evaluate_predicted_frames(references, {})

    assert result[0].predicted_frame is None
    assert result[0].within_tolerance is False


def test_duplicate_gate_ids_are_rejected():
    with pytest.raises(ValueError, match="doppelte"):
        FrameReferenceDocument(
            "run.mp4",
            2,
            [GateFrameReference(1, 70), GateFrameReference(1, 71)],
        )
