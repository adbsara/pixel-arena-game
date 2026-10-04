"""Export the trained network's weights for PyTorch-free play."""

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from pixel_arena.envs import ChaseEnv
from pixel_arena.game.policy import NumpyPolicy
from pixel_arena.training.evaluate import evaluate

MODEL_PATH = Path("models/chaser.zip")
WEIGHTS_PATH = Path("models/chaser.npz")

# The layers that turn an observation into action scores, in order,
# under the names Stable-Baselines3 gives them.
LAYERS = [
    "mlp_extractor.policy_net.0",
    "mlp_extractor.policy_net.2",
    "action_net",
]


def main() -> None:
    model = PPO.load(MODEL_PATH, device="cpu")
    state = model.policy.state_dict()

    arrays = {}
    for i, name in enumerate(LAYERS):
        arrays[f"w{i}"] = state[f"{name}.weight"].detach().cpu().numpy()
        arrays[f"b{i}"] = state[f"{name}.bias"].detach().cpu().numpy()
    np.savez(WEIGHTS_PATH, **arrays)
    print(f"weights saved to {WEIGHTS_PATH}")

    policy = NumpyPolicy.load(WEIGHTS_PATH)

    # Check 1: the numpy network picks the same action as the original.
    space = ChaseEnv().observation_space
    space.seed(0)
    total = 1000
    matches = 0
    for _ in range(total):
        obs = space.sample()
        expected, _ = model.predict(obs, deterministic=True)
        matches += int(policy(obs) == int(expected))
    print(f"same action on {matches}/{total} observations")

    # Check 2: it plays as well as the original on the same 200 episodes.
    result = evaluate(policy)
    print("numpy enemy:", {k: round(v, 3) for k, v in result.items()})


if __name__ == "__main__":
    main()