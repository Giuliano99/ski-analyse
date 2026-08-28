import numpy as np
import pytest

from src.detection.gate_detector import GateDetector


class FakeModel:
    def __init__(self):
        self.call = None

    def infer(self, image, *, model_id):
        self.call = (image, model_id)
        return {
            "predictions": [
                {
                    "x": 100,
                    "y": 80,
                    "width": 40,
                    "height": 60,
                    "confidence": 0.91,
                    "class": "Ski-Gates",
                },
                {
                    "x": 20,
                    "y": 20,
                    "width": 10,
                    "height": 10,
                    "confidence": 0.2,
                    "class": "Ski-Gates",
                },
                {
                    "x": 101,
                    "y": 81,
                    "width": 40,
                    "height": 60,
                    "confidence": 0.7,
                    "class": "Ski-Gates",
                },
                {
                    "x": 100,
                    "y": 80,
                    "width": 18,
                    "height": 20,
                    "confidence": 0.99,
                    "class": "Ski-Gates",
                },
                {
                    "x": 20,
                    "y": 20,
                    "width": 10,
                    "height": 10,
                    "confidence": 0.8,
                    "class": "andere-klasse",
                },
            ]
        }
def test_detector_maps_roboflow_prediction_and_filters_class():
    model = FakeModel()
    detector = GateDetector(model, model_id="workspace/model", confidence=0.35, overlap=0.5)
    image = np.zeros((120, 160, 3), dtype=np.uint8)

    detections = detector.detect(image)

    assert model.call[1] == "workspace/model"
    assert len(detections) == 1
    assert detections[0].xyxy == (80, 50, 120, 110)
    assert detections[0].confidence == pytest.approx(0.91)


def test_detector_rejects_invalid_thresholds_and_grayscale_images():
    model = FakeModel()
    with pytest.raises(ValueError, match="confidence"):
        GateDetector(model, model_id="workspace/model", confidence=1.1)

    detector = GateDetector(model, model_id="workspace/model")
    with pytest.raises(ValueError, match="BGR"):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))
