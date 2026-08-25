from pathlib import Path

from src.timing.config import load_timing_config
from src.timing.lap_time import StartMode


def test_template_uses_first_gate_as_start():
    config_path = Path(__file__).parents[1] / "configs" / "gates_template.yaml"

    config = load_timing_config(config_path)

    assert config.start_mode == StartMode.FIRST_GATE
    assert config.start_gate_id == 1
    assert config.video_fps == 50.0


def test_manual_frame_config(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "video_fps: 25\n"
        "timing:\n"
        "  start_mode: manual_frame\n"
        "  manual_start_frame: 125\n",
        encoding="utf-8",
    )

    config = load_timing_config(config_path)

    assert config.start_mode == StartMode.MANUAL_FRAME
    assert config.manual_start_frame == 125
