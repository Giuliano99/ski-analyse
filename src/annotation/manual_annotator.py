"""Interaktives OpenCV Werkzeug fuer manuelle Referenzdurchfahrten."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.annotation.manual_annotations import (
    FrameAnnotation,
    ManualAnnotationDocument,
    ManualGateCrossing,
    load_document,
    save_document,
)
from src.io.video_reader import VideoReader
from src.timing.geometry import Point, moving_gate_crossing_fraction


POINT_NAMES = ("Athlet", "Tor links", "Tor rechts")
POINT_COLORS = ((0, 255, 255), (0, 255, 0), (0, 0, 255))


class ManualAnnotator:
    def __init__(
        self,
        video_path: Path,
        output_path: Path,
        target_athlete: str,
        allow_uncertain: bool = False,
    ):
        self.video_path = video_path
        self.output_path = output_path
        self.allow_uncertain = allow_uncertain
        with VideoReader(video_path) as reader:
            metadata = reader.metadata

        self.capture = cv2.VideoCapture(str(video_path))
        self.frame_count = metadata.frame_count
        self.width = metadata.width
        self.height = metadata.height
        if output_path.exists():
            self.document = load_document(output_path)
            if self.document.video_name != video_path.name:
                raise ValueError("Annotationsdatei gehoert zu einem anderen Video")
        else:
            self.document = ManualAnnotationDocument(
                video_name=video_path.name,
                width=self.width,
                height=self.height,
                fps=metadata.fps,
                target_athlete=target_athlete,
            )

        self.frame_number = 0
        self.frame = None
        self.timestamp_s = 0.0
        self.scale = min(1280 / self.width, 720 / self.height, 1.0)
        self.mode: str | None = None
        self.points: list[Point] = []
        self.previous: FrameAnnotation | None = None
        self.message = "N startet eine neue Tordurchfahrt"
        self._read_frame(0)

    def _read_frame(self, frame_number: int) -> None:
        frame_number = max(0, min(self.frame_count - 1, frame_number))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()
        if not ok:
            raise ValueError(f"Frame {frame_number} kann nicht gelesen werden")
        if self.mode == "current" and self.points:
            self.points.clear()
        self.frame_number = frame_number
        self.frame = frame
        timestamp_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
        self.timestamp_s = timestamp_ms / 1000.0

    def _frame_annotation(self) -> FrameAnnotation:
        return FrameAnnotation(
            frame_number=self.frame_number,
            timestamp_s=self.timestamp_s,
            athlete_pos=self.points[0],
            gate_left=self.points[1],
            gate_right=self.points[2],
        )

    def _next_gate_id(self) -> int:
        return max((item.gate_id for item in self.document.crossings), default=0) + 1

    def _mouse(self, event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.message = "Letzten Punkt entfernt"
            return
        if event != cv2.EVENT_LBUTTONDOWN or self.mode is None:
            return

        self.points.append(Point(x / self.scale, y / self.scale))
        if len(self.points) < 3:
            self.message = f"Jetzt markieren: {POINT_NAMES[len(self.points)]}"
            return

        annotation = self._frame_annotation()
        self.points.clear()
        if self.mode == "previous":
            self.previous = annotation
            self.mode = "current"
            self.message = "Zu einem Frame nach der Durchfahrt wechseln und 3 Punkte markieren"
            return

        assert self.previous is not None
        if annotation.frame_number <= self.previous.frame_number:
            self.message = "Der zweite Frame muss spaeter sein. Punkte erneut markieren"
            return
        crossing = ManualGateCrossing(
            gate_id=self._next_gate_id(),
            previous=self.previous,
            current=annotation,
        )
        fraction = moving_gate_crossing_fraction(
            crossing.previous.athlete_pos,
            crossing.current.athlete_pos,
            crossing.previous.gate_left,
            crossing.previous.gate_right,
            crossing.current.gate_left,
            crossing.current.gate_right,
        )
        if fraction is None and not self.allow_uncertain:
            self.mode = "current"
            self.message = (
                "Kein Seitenwechsel: anderen Frame nach der Durchfahrt waehlen "
                "und die 3 Punkte erneut markieren"
            )
            return

        self.document.add_crossing(crossing)
        save_document(self.document, self.output_path)
        self.mode = None
        self.previous = None
        if fraction is None:
            self.message = (
                f"Tor {crossing.gate_id} gespeichert, aber ohne Seitenwechsel (unsicher)"
            )
        else:
            self.message = f"Tor {crossing.gate_id} gespeichert. N fuer das naechste Tor"

    def _draw_annotation(self, canvas, annotation: FrameAnnotation) -> None:
        points = (annotation.athlete_pos, annotation.gate_left, annotation.gate_right)
        for point, color in zip(points, POINT_COLORS):
            cv2.circle(
                canvas,
                (round(point.x * self.scale), round(point.y * self.scale)),
                7,
                color,
                -1,
            )

    def _render(self):
        assert self.frame is not None
        canvas = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale)
        for crossing in self.document.crossings:
            if crossing.previous.frame_number == self.frame_number:
                self._draw_annotation(canvas, crossing.previous)
            if crossing.current.frame_number == self.frame_number:
                self._draw_annotation(canvas, crossing.current)
        for point, color in zip(self.points, POINT_COLORS):
            cv2.circle(
                canvas,
                (round(point.x * self.scale), round(point.y * self.scale)),
                7,
                color,
                -1,
            )

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 92), (0, 0, 0), -1)
        lines = (
            f"Frame {self.frame_number}/{self.frame_count - 1}  Zeit {self.timestamp_s:.3f}s  "
            f"Tore {len(self.document.crossings)}",
            "A/D: 1 Frame  J/L: 5 Frames  U/O: 30 Frames  N: neues Tor  Rechtsklick: zurueck",
            f"Z: Abbruch  X: letztes Tor loeschen  S: speichern  Q: speichern und beenden | {self.message}",
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

    def run(self) -> None:
        window = "Ski Analyse - manuelle Tordurchfahrten"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, self._mouse)
        try:
            while True:
                cv2.imshow(window, self._render())
                key = cv2.waitKeyEx(0)
                if key in (ord("q"), ord("Q"), 27):
                    save_document(self.document, self.output_path)
                    break
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
                elif key in (ord("n"), ord("N")):
                    self.mode = "previous"
                    self.points.clear()
                    self.previous = None
                    self.message = "Vorher Frame: Athlet, Tor links, Tor rechts markieren"
                elif key in (ord("z"), ord("Z")):
                    self.mode = None
                    self.points.clear()
                    self.previous = None
                    self.message = "Aktuelle Markierung abgebrochen"
                elif key in (ord("x"), ord("X")) and self.document.crossings:
                    deleted = self.document.crossings.pop()
                    save_document(self.document, self.output_path)
                    self.message = f"Tor {deleted.gate_id} entfernt"
                elif key in (ord("s"), ord("S")):
                    save_document(self.document, self.output_path)
                    self.message = "Gespeichert"
        finally:
            self.capture.release()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--athlete", default="bestaetigter Zielathlet")
    parser.add_argument(
        "--allow-uncertain",
        action="store_true",
        help="Markierungen ohne geometrischen Seitenwechsel bewusst speichern",
    )
    args = parser.parse_args()
    output = args.output or Path("data/manual_annotations") / f"{args.video.stem}.json"
    ManualAnnotator(args.video, output, args.athlete, args.allow_uncertain).run()


if __name__ == "__main__":
    main()
