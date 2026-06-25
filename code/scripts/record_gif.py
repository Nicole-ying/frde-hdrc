"""Record GIF of trained FDRE-HRDC agent on LunarLander-v3."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO


def record_gif(model_path: str, output_path: str, episodes: int = 3, fps: int = 30):
    """Load model and record gameplay GIF on official LunarLander-v3."""
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    all_frames = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            step += 1

            frame = env.render()
            all_frames.append(frame)

        print(f"Episode {ep + 1}: score={total_reward:.1f}, steps={step}")

    env.close()

    # 存 GIF（隔帧采样控制大小）
    sample_every = max(1, fps // 15)
    sampled = all_frames[::sample_every]
    imageio.mimsave(output_path, sampled, fps=15, loop=0)
    print(f"GIF saved: {output_path} ({len(sampled)} frames)")


if __name__ == "__main__":
    model_path = ROOT / "outputs/lunarlander_3seed_8iter_1M/seed_43/model_best.zip"
    output_path = ROOT / "outputs/lunarlander_3seed_8iter_1M/seed_43/gameplay.gif"
    record_gif(str(model_path), str(output_path), episodes=3)
