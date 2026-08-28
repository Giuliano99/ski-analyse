"""Stichprobenartige Torerkennung in einem Video ausfuehren."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import yaml
from dotenv import load_dotenv

from src.detection.gate_detector import GateDetection, GateDetector
from src.io.video_reader import VideoReader


def draw_detections(image, detections: list[GateDetection]):
    preview = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"Tor {detection.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return preview


def load_detector(config_path: Path) -> GateDetector:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    roboflow = config["roboflow"]
    return GateDetector.from_roboflow(
        model_id=roboflow["model_id"],
        confidence=float(config["inference"]["confidence"]),
        overlap=float(config["inference"]["overlap"]),
    )


def analyze_video(
    video_path: Path,
    detector: GateDetector,
    output_dir: Path,
    *,
    sample_seconds: float,
    max_samples: int | None,
) -> Path:
    if sample_seconds <= 0:
        raise ValueError("sample_seconds muss groesser als 0 sein")

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []

    with VideoReader(video_path) as reader:
        every_n_frames = max(1, round(reader.metadata.fps * sample_seconds))
        metadata = {
            "video": str(video_path),
            "fps": reader.metadata.fps,
            "frame_count": reader.metadata.frame_count,
            "sample_seconds": sample_seconds,
        }
        for frame in reader:
            if frame.frame_number % every_n_frames != 0:
                continue
            detections = detector.detect(frame.image)
            records.append(
                {
                    "frame_number": frame.frame_number,
                    "timestamp_s": frame.timestamp_s,
                    "detections": [asdict(detection) for detection in detections],
                }
            )
            preview = draw_detections(frame.image, detections)
            cv2.imwrite(str(preview_dir / f"frame_{frame.frame_number:06d}.jpg"), preview)
            if max_samples is not None and len(records) >= max_samples:
                break

    result_path = output_dir / "detections.json"
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump({"metadata": metadata, "frames": records}, result_file, indent=2)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    load_dotenv()
    detector = load_detector(args.config)
    output_dir = args.output_dir or Path("outputs/gate_detection") / args.video.stem
    result_path = analyze_video(
        args.video,
        detector,
        output_dir,
        sample_seconds=args.sample_seconds,
        max_samples=args.max_samples,
    )
    print(f"Ergebnis gespeichert: {result_path}")


if __name__ == "__main__":
    main()
