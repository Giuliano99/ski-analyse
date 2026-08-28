"""Manuelle Referenzannotation in eine Torzeiten CSV umwandeln."""

import argparse
from pathlib import Path

from src.annotation.manual_annotations import load_document, to_observations
from src.timing.lap_time import (
    StartMode,
    TimingConfig,
    build_gate_crossing_events,
    write_gate_events_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument(
        "--start-mode",
        choices=[mode.value for mode in StartMode],
        default=StartMode.FIRST_GATE.value,
    )
    parser.add_argument("--manual-start-frame", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    document = load_document(args.annotation)
    config = TimingConfig(
        start_mode=StartMode(args.start_mode),
        video_fps=document.fps,
        manual_start_frame=args.manual_start_frame,
    )
    events = build_gate_crossing_events(to_observations(document), config)
    output = args.output or Path("outputs") / f"{Path(document.video_name).stem}_times.csv"
    write_gate_events_csv(events, output)
    uncertain_count = sum(event.uncertain for event in events)
    print(
        f"{len(events)} Torzeiten gespeichert: {output} "
        f"({uncertain_count} als unsicher markiert)"
    )


if __name__ == "__main__":
    main()
