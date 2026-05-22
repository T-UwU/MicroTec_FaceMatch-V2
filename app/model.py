"""ONNX model inference."""

import onnxruntime as ort
import numpy as np
from pathlib import Path


class AntiSpoofingModel:
    def __init__(self, model_path: str = None):
        model_dir = Path(__file__).parent.parent / "model"
        if model_path is None:
            model_path = str(model_dir / "model.onnx")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.threshold = 0.5

        # Load model version
        version_file = model_dir / "version.txt"
        self.version = version_file.read_text().strip() if version_file.exists() else "unknown"

    def predict(self, frames: np.ndarray, landmarks: np.ndarray) -> dict:
        """Run inference. Returns decision dict."""
        inputs = {
            self.input_names[0]: frames,
            self.input_names[1]: landmarks,
        }
        output = self.session.run(None, inputs)
        score = float(output[0].item())

        return {
            "decision": "SPOOF" if score > self.threshold else "REAL",
            "score": round(score, 6),
            "threshold": self.threshold,
        }

