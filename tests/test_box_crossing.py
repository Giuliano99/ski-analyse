import pytest

from src.timing.geometry import BoundingBox, moving_box_gate_crossing


def test_box_crossing_uses_multiple_lower_edge_points():
    result = moving_box_gate_crossing(
        BoundingBox(2, -4, 8, -2),
        BoundingBox(2, 0, 8, 2),
        BoundingBox(0, -1, 10, 0),
        BoundingBox(0, -1, 10, 0),
    )

    assert result is not None
    assert result.fraction == pytest.approx(0.5)
    assert result.supporting_points == 3
    assert result.uncertain is False


def test_box_edge_can_cross_when_old_center_point_is_outside_gate():
    result = moving_box_gate_crossing(
        BoundingBox(8, -4, 14, -2),
        BoundingBox(8, 0, 14, 2),
        BoundingBox(0, -1, 10, 0),
        BoundingBox(0, -1, 10, 0),
    )

    assert result is not None
    assert result.supporting_points == 1
    assert result.uncertain is True


def test_equal_camera_motion_does_not_create_box_crossing():
    result = moving_box_gate_crossing(
        BoundingBox(2, -4, 8, -2),
        BoundingBox(5, -4, 11, -2),
        BoundingBox(0, -1, 10, 0),
        BoundingBox(3, -1, 13, 0),
    )

    assert result is None


def test_invalid_horizontal_sample_is_rejected():
    with pytest.raises(ValueError, match="horizontal_samples"):
        moving_box_gate_crossing(
            BoundingBox(2, -4, 8, -2),
            BoundingBox(2, 0, 8, 2),
            BoundingBox(0, -1, 10, 0),
            BoundingBox(0, -1, 10, 0),
            horizontal_samples=(-0.1,),
        )
