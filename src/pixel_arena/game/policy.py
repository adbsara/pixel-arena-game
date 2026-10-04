"""The enemy's brain at play time: a tiny neural network in pure numpy."""

from pathlib import Path

import numpy as np


class NumpyPolicy:
    """Runs the trained network without PyTorch."""

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray]):
        self.weights = weights
        self.biases = biases

    @classmethod
    def load(cls, path: str | Path) -> "NumpyPolicy":
        """Read the layers from a .npz file written by the export script."""
        data = np.load(path)
        n_layers = len(data.files) // 2
        return cls(
            [data[f"w{i}"] for i in range(n_layers)],
            [data[f"b{i}"] for i in range(n_layers)],
        )

    def __call__(self, obs: np.ndarray) -> int:
        """Map an observation to the best action the network knows."""
        x = obs
        # Hidden layers: multiply, add, squash with tanh.
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(w @ x + b)
        # Last layer: one score per action. Pick the highest.
        scores = self.weights[-1] @ x + self.biases[-1]
        return int(np.argmax(scores))