"""Train the enemy to catch the player with PPO."""

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from pixel_arena.envs import ChaseEnv
from pixel_arena.training.evaluate import evaluate

N_ENVS = 8                  # copies of the game running in parallel
TOTAL_TIMESTEPS = 500_000   # total frames of experience to collect
SEED = 0

MODEL_PATH = Path("models/chaser")
LOG_DIR = "runs"


def main() -> None:
    env = make_vec_env(ChaseEnv, n_envs=N_ENVS, seed=SEED)

    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cpu",  # a network this small trains faster on CPU
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"model saved to {MODEL_PATH}.zip")

    def trained_policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    # Same 200 starting positions as the random baseline.
    result = evaluate(trained_policy)
    print("trained enemy:", {k: round(v, 3) for k, v in result.items()})


if __name__ == "__main__":
    main()