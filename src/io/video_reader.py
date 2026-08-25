"""Video-Metadaten und Frames mit Zeitstempeln einlesen."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> Optional[float]:
        if self.fps <= 0 or self.frame_count <= 0:
            return None
        return self.frame_count / self.fps


@dataclass(frozen=True)
class VideoFrame:
    frame_number: int
    timestamp_s: float
    image: np.ndarray


class VideoReader:
    """Iteriert genau einmal ueber ein Video und gibt die Ressource frei."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video nicht gefunden: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"Video kann nicht geoeffnet werden: {self.path}")

        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            self._capture.release()
            raise ValueError(f"Ungueltige Framerate fuer Video: {self.path}")

        self.metadata = VideoMetadata(
            path=self.path,
            fps=fps,
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        self._closed = False

    def __iter__(self) -> Iterator[VideoFrame]:
        frame_number = 0
        while not self._closed:
            ok, image = self._capture.read()
            if not ok:
                break

            timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_s = (
                timestamp_ms / 1000.0
                if timestamp_ms > 0 or frame_number == 0
                else frame_number / self.metadata.fps
            )
            yield VideoFrame(frame_number, timestamp_s, image)
            frame_number += 1

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
