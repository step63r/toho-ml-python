import gymnasium as gym
import numpy as np

from gymnasium import spaces
from typing import SupportsFloat


class ChannelLast(gym.ObservationWrapper):
    """グレー画像 (高さ, 幅) を (高さ, 幅, 1) にする。SB3 の CnnPolicy と VecFrameStack が、チャンネルの軸を必要とするため"""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        space = env.observation_space
        assert isinstance(space, spaces.Box) and len(space.shape) == 2, "2 次元のグレー画像だけ扱える"
        self.observation_space = spaces.Box(low=0, high=255, shape=space.shape + (1,), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[..., np.newaxis]


class ScaledReward(gym.RewardWrapper):
    """
    報酬を一定倍する。環境の報酬は、ゲームオーバー -100・チャプター +100・スペルカード +500・ステージ +1000 と、
    アイテム取得の小さな値が混ざる。Q 値の大きさを抑えるために縮める (符号と比率は変えない。クリップはしない)
    """
    def __init__(self, env: gym.Env, scale: float = 0.01) -> None:
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: SupportsFloat) -> float:
        return float(reward) * self.scale
