import pytest

from src.timing.geometry import (
    Point,
    gate_crossed,
    moving_gate_crossing_fraction,
    segments_intersect,
)


def test_athlete_crosses_gate_line():
    gate_left = Point(0, 0)
    gate_right = Point(10, 0)

    prev_pos = Point(5, -2)
    curr_pos = Point(5, 2)

    assert gate_crossed(prev_pos, curr_pos, gate_left, gate_right) is True


def test_athlete_moves_parallel_no_crossing():
    gate_left = Point(0, 0)
    gate_right = Point(10, 0)

    prev_pos = Point(5, -2)
    curr_pos = Point(6, -1)

    assert gate_crossed(prev_pos, curr_pos, gate_left, gate_right) is False


def test_crossing_outside_gate_segment_not_counted():
    gate_left = Point(0, 0)
    gate_right = Point(10, 0)

    # Bewegung kreuzt die verlaengerte Linie, aber ausserhalb der Torbreite
    prev_pos = Point(20, -2)
    curr_pos = Point(20, 2)

    assert gate_crossed(prev_pos, curr_pos, gate_left, gate_right) is False


def test_segments_intersect_basic_cross():
    p1, p2 = Point(0, 0), Point(4, 4)
    p3, p4 = Point(0, 4), Point(4, 0)

    assert segments_intersect(p1, p2, p3, p4) is True


def test_moving_camera_without_relative_crossing_is_not_counted():
    fraction = moving_gate_crossing_fraction(
        Point(5, -2), Point(8, -2),
        Point(0, 0), Point(10, 0),
        Point(3, 0), Point(13, 0),
    )

    assert fraction is None


def test_relative_crossing_with_moving_camera_is_interpolated():
    fraction = moving_gate_crossing_fraction(
        Point(5, -2), Point(8, 2),
        Point(0, 0), Point(10, 0),
        Point(3, 0), Point(13, 0),
    )

    assert fraction == pytest.approx(0.5)


def test_relative_crossing_outside_moving_gate_is_not_counted():
    fraction = moving_gate_crossing_fraction(
        Point(20, -2), Point(23, 2),
        Point(0, 0), Point(10, 0),
        Point(3, 0), Point(13, 0),
    )

    assert fraction is None
