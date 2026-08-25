"""Laden und Validieren der Zeitmessung aus einer Laufkonfiguration."""

from pathlib import Path

import yaml

from src.timing.lap_time import StartMode, TimingConfig


def load_timing_config(path: str | Path) -> TimingConfig:
    source = Path(path)
    with source.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)

    if not isinstance(data, dict) or not isinstance(data.get("timing"), dict):
        raise ValueError(f"Abschnitt 'timing' fehlt in {source}")

    timing = data["timing"]
    try:
        start_mode = StartMode(timing["start_mode"])
        video_fps = float(data["video_fps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Ungueltige Zeitkonfiguration in {source}: {error}") from error

    return TimingConfig(
        start_mode=start_mode,
        video_fps=video_fps,
        start_gate_id=int(timing.get("start_gate_id", 1)),
        manual_start_frame=timing.get("manual_start_frame"),
    )
