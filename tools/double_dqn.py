"""
Double DQN

SB3 の DQN は、次の状態の価値を「目標ネットワークの最大値」にする。これは過大評価しやすい。
Double DQN は、行動の選択を本体のネットワークで、その価値の評価を目標ネットワークで行い、過大評価を抑える。

実機の学習 (約 9 万ステップ) で、報酬は死亡の -1 と、まれな +1 しかないのに、Q 値が +0.76 まで上がり (真の値は約 -0.76)、
学習が進むほど真の値との相関が下がった (0.47 → 0.28)。保存したデータでの比較で、Double・γ の引き下げ・n ステップが、
この過大評価を抑えた (CLAUDE.md を参照)。
"""
import numpy as np
import torch as th

from stable_baselines3 import DQN
from torch.nn import functional as F
from typing import Callable


def double_q_target(q_net: Callable[[th.Tensor], th.Tensor], q_net_target: Callable[[th.Tensor], th.Tensor],
                    next_observations: th.Tensor, rewards: th.Tensor, dones: th.Tensor, discounts) -> th.Tensor:
    """
    Double DQN の目標値: r + (1 - done) * discount * Q_target(s', argmax_a Q(s', a))

    discounts は、n ステップでは打ち切りに応じて変わる割引 (配列)、1 ステップでは γ (数値)。
    """
    with th.no_grad():
        next_actions = q_net(next_observations).argmax(dim=1, keepdim=True)
        next_q_values = q_net_target(next_observations).gather(1, next_actions)
        return rewards + (1 - dones) * discounts * next_q_values


class DoubleDQN(DQN):
    """DQN の目標値を Double DQN にしたもの (他は SB3 の DQN と同じ。n ステップのバッファにも対応)"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 保存したモデルが Double かどうかを、読み込む前に判別できるように、属性として保存させる (saved_is_double)
        self.double = True

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            target_q_values = double_q_target(self.q_net, self.q_net_target, replay_data.next_observations,
                                              replay_data.rewards, replay_data.dones, discounts)

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
