"""Athletenbox in einem geeigneten Startframe manuell auswaehlen."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.io.video_reader import VideoReader
from src.tracking.athlete_tracker import AthleteSelection, save_selection


class AthleteSelector:
    def __init__(self, video_path: Path, output_path: Path):
        self.video_path = video_path
        self.output_path = output_path
        with VideoReader(video_path) as reader:
            self.metadata = reader.metadata

        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise ValueError(f"Video kann nicht geoeffnet werden: {video_path}")
        self.frame_number = 0
        self.timestamp_s = 0.0
        self.frame = None
        self.scale = min(1280 / self.metadata.width, 720 / self.metadata.height, 1.0)
        self._read_frame(0)

    def _read_frame(self, frame_number: int) -> None:
        frame_number = max(0, min(self.metadata.frame_count - 1, frame_number))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()
        if not ok:
            raise ValueError(f"Frame {frame_number} kann nicht gelesen werden")
        self.frame_number = frame_number
        self.frame = frame
        timestamp_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
        self.timestamp_s = timestamp_ms / 1000.0

    def _render(self):
        canvas = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 66), (0, 0, 0), -1)
        lines = (
            f"Frame {self.frame_number}/{self.metadata.frame_count - 1}  "
            f"Zeit {self.timestamp_s:.3f}s",
            "A/D: 1 Frame  J/L: 5  U/O: 30  B: Athletenbox ziehen  Q: beenden",
        )
        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (12, 24 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _select_box(self, window: str) -> AthleteSelection | None:
        canvas = self._render()
        x, y, width, height = cv2.selectROI(
            window,
            canvas,
            showCrosshair=True,
            fromCenter=False,
        )
        if width <= 0 or height <= 0:
            return None
        return AthleteSelection(
            video_name=self.video_path.name,
            frame_number=self.frame_number,
            timestamp_s=self.timestamp_s,
            x=x / self.scale,
            y=y / self.scale,
            width=width / self.scale,
            height=height / self.scale,
        )

    def run(self) -> AthleteSelection | None:
        window = "Ski Analyse - Athlet auswaehlen"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        try:
            while True:
                cv2.imshow(window, self._render())
                key = cv2.waitKeyEx(0)
                if key in (ord("q"), ord("Q"), 27):
                    return None
                if key in (ord("a"), ord("A"), 2424832):
                    self._read_frame(self.frame_number - 1)
                elif key in (ord("d"), ord("D"), 2555904):
                    self._read_frame(self.frame_number + 1)
                elif key in (ord("j"), ord("J")):
                    self._read_frame(self.frame_number - 5)
                elif key in (ord("l"), ord("L")):
                    self._read_frame(self.frame_number + 5)
                elif key in (ord("u"), ord("U")):
                    self._read_frame(self.frame_number - 30)
                elif key in (ord("o"), ord("O")):
                    self._read_frame(self.frame_number + 30)
                elif key in (ord("b"), ord("B"), 13, 32):
                    selection = self._select_box(window)
                    if selection is not None:
                        save_selection(selection, self.output_path)
                        return selection
        finally:
            self.capture.release()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("data/athlete_selections") / f"{args.video.stem}.json"
    selection = AthleteSelector(args.video, output).run()
    if selection is None:
        print("Keine Athletenbox gespeichert.")
        return
    print(
        f"Athletenbox gespeichert: {output} "
        f"(Frame {selection.frame_number}, Zeit {selection.timestamp_s:.3f}s)"
    )


if __name__ == "__main__":
    main()
