"""Athleten ab einer gespeicherten Startbox automatisch verfolgen."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tracking.athlete_tracker import load_selection, track_athlete, track_athlete_hybrid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preview-every", type=int, default=30)
    parser.add_argument("--method", choices=("hybrid", "csrt", "mil"), default="hybrid")
    parser.add_argument("--person-model", default="yolo11n.pt")
    args = parser.parse_args()

    selection_path = args.selection or Path("data/athlete_selections") / f"{args.video.stem}.json"
    output_dir = args.output_dir or Path("outputs/athlete_tracking") / args.video.stem
    selection = load_selection(selection_path)
    tracking_function = track_athlete_hybrid if args.method == "hybrid" else track_athlete
    tracking_options = {"preview_every": args.preview_every}
    if args.method == "hybrid":
        tracking_options["model_path"] = args.person_model
    else:
        tracking_options["tracker_type"] = args.method
    result = tracking_function(
        args.video,
        selection,
        output_dir / "tracking.json",
        output_dir / "previews",
        **tracking_options,
    )
    print(f"Tracking gespeichert: {result}")


if __name__ == "__main__":
    main()
