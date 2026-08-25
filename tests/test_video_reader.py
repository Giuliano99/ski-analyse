from pathlib import Path

import cv2
import numpy as np
import pytest

from src.io.video_reader import VideoReader


def create_test_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("MJPG VideoWriter ist in dieser OpenCV Installation nicht verfuegbar")
    for value in range(frame_count):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()


def test_video_reader_returns_metadata_frames_and_timestamps(tmp_path):
    path = tmp_path / "sample.avi"
    create_test_video(path)

    with VideoReader(path) as reader:
        frames = list(reader)
        metadata = reader.metadata

    assert metadata.fps == pytest.approx(10.0)
    assert (metadata.width, metadata.height, metadata.frame_count) == (32, 24, 3)
    assert metadata.duration_s == pytest.approx(0.3)
    assert [frame.frame_number for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_s for frame in frames] == pytest.approx([0.0, 0.1, 0.2])


def test_video_reader_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Video nicht gefunden"):
        VideoReader(tmp_path / "missing.mp4")
