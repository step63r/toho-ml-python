"""
探索 (ε-greedy のランダムな行動) の分布を、人のプレイの傾向に差し替える

SB3 の DQN は、探索のランダムな行動を action_space.sample() で引く。一様ランダムは、常に動き回って、すぐ死ぬ
(同じ状態での評価で、中央値 47 ステップ)。人のプレイの記録では、半分の時間は方向を押していない (止まっている)。
行動の割合を人に合わせ、前の行動を一定の確率で続ける (人は、キーを押しっぱなしにする) と、中央値 67.5・平均 90 ステップ、
最大 253 ステップまで生きた (止まり続ける方策は、中央値 72)。探索で、エピソードの後ろの場面のデータが増える。
"""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from gymnasium import spaces


class PriorDiscrete(spaces.Discrete):
    """
    sample() が、指定した割合で行動を引く Discrete。sticky の確率で、前に引いた行動を続ける

    SB3 のモデルに、行動空間として差し替えて使う (環境の行動空間は、そのまま)。
    """
    def __init__(self, probabilities: Sequence[float], sticky: float = 0.0, seed: Optional[int] = None) -> None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        super().__init__(len(probabilities))
        if probabilities.min() < 0 or probabilities.sum() <= 0:
            raise ValueError("行動の割合は、0 以上で、合計が正でなければなりません。")
        if not 0.0 <= sticky <= 1.0:
            raise ValueError("sticky は 0〜1 にしてください。")
        self.probabilities = probabilities / probabilities.sum()
        self.sticky = sticky
        self._rng = np.random.default_rng(seed)
        self._last: Optional[int] = None

    def sample(self, mask=None, probability=None) -> np.int64:
        if self._last is None or self._rng.random() >= self.sticky:
            self._last = int(self._rng.choice(self.n, p=self.probabilities))
        return np.int64(self._last)


def human_prior_space(session_dirs: Sequence[Path], sticky: float, seed: Optional[int] = None) -> PriorDiscrete:
    """人のプレイの記録 (プレイ中のフレーム) での行動の割合から、探索の行動空間を作る"""
    from tools.evaluate_dqn import action_prior
    return PriorDiscrete(action_prior(session_dirs), sticky, seed)


def set_exploration(model, space: Optional[PriorDiscrete]) -> None:
    """
    モデルの探索の分布を設定する。space が None なら、一様ランダム (環境の行動空間と同じ Discrete) に戻す

    差し替えは、モデルの保存に含まれる。再開のたびに、指定どおりに設定し直すこと (指定しなければ、一様ランダムに戻る)。
    """
    model.action_space = space if space is not None else spaces.Discrete(model.action_space.n)
