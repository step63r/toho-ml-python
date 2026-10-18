import argparse
import json
import zipfile

import numpy as np
import pytest
import torch as th

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import NStepReplayBuffer

from tests.test_train_dqn import FakeGameEnv, PARAMS, fill
from tools import train_dqn
from tools.double_dqn import DoubleDQN, double_q_target


def test_Double_DQNの目標値は_行動を本体で選び_価値を目標ネットワークで評価する():
    q_net = lambda obs: th.tensor([[1.0, 5.0]])             # 本体は、行動 1 を選ぶ
    q_target = lambda obs: th.tensor([[10.0, 3.0]])         # 目標ネットワークの最大 (行動 0) は 10。行動 1 の価値は 3

    target = double_q_target(q_net, q_target, th.zeros(1, 1), th.tensor([[1.0]]), th.tensor([[0.0]]), 0.9)

    assert target.item() == pytest.approx(1.0 + 0.9 * 3.0)       # 通常の DQN なら 1 + 0.9 * 10


def test_終了した遷移は_次の状態の価値を使わない():
    q_net = lambda obs: th.tensor([[1.0, 5.0]])
    q_target = lambda obs: th.tensor([[10.0, 3.0]])

    target = double_q_target(q_net, q_target, th.zeros(1, 1), th.tensor([[-1.0]]), th.tensor([[1.0]]), 0.9)

    assert target.item() == pytest.approx(-1.0)


def test_n_stepの割引は_遷移ごとの値が使われる():
    q_net = lambda obs: th.tensor([[1.0, 5.0], [1.0, 5.0]])
    q_target = lambda obs: th.tensor([[10.0, 2.0], [10.0, 2.0]])
    discounts = th.tensor([[0.5], [0.25]])                  # 打ち切りで、遷移ごとに割引が違う

    target = double_q_target(q_net, q_target, th.zeros(2, 1), th.zeros(2, 1), th.zeros(2, 1), discounts)

    assert target.flatten().tolist() == pytest.approx([1.0, 0.5])


def make_double(tmp_path, n_steps=1, double=True):
    venv = train_dqn.build_vec_env(lambda: FakeGameEnv(), tmp_path, n_stack=4)
    params = dict(PARAMS, n_steps=n_steps, gamma=0.95)
    return train_dqn.build_model(venv, tmp_path, params, double=double), venv


def test_DoubleDQNは学習でき_nステップのバッファを使える(tmp_path):
    model, venv = make_double(tmp_path, n_steps=3)

    result = train_dqn.train(model, tmp_path, total_steps=150, checkpoint_every=1000)

    assert result == "final"
    assert isinstance(model, DoubleDQN) and isinstance(model.replay_buffer, NStepReplayBuffer)
    assert model.replay_buffer.next_observations is not None        # メモリ節約モードではない
    assert model._n_updates > 0


def test_保存したモデルが_Doubleかどうかを読み込む前に判別し_再開でも同じクラスで読む(tmp_path):
    double_model, venv = make_double(tmp_path / "d", n_steps=3)
    plain_model, venv2 = make_double(tmp_path / "p", n_steps=1, double=False)
    train_dqn.train(double_model, tmp_path / "d", total_steps=60, checkpoint_every=1000)
    train_dqn.train(plain_model, tmp_path / "p", total_steps=60, checkpoint_every=1000)

    assert train_dqn.saved_is_double(tmp_path / "d" / "final.zip") is True
    assert train_dqn.saved_is_double(tmp_path / "p" / "final.zip") is False
    venv3 = train_dqn.build_vec_env(lambda: FakeGameEnv(), tmp_path / "d", n_stack=4)
    resumed = train_dqn.load_for_resume(tmp_path / "d" / "final.zip", venv3, tmp_path / "d", device="cpu")
    assert isinstance(resumed, DoubleDQN) and resumed.n_steps == 3
    assert resumed.replay_buffer.size() == double_model.replay_buffer.size()        # 次の観測も含めて読み戻せる


def test_nステップのバッファは_次の観測も保存して読み戻せる(tmp_path):
    model, venv = make_double(tmp_path / "a", n_steps=3)
    fill(model, 20)
    train_dqn.save_replay_buffer(model.replay_buffer, tmp_path / "buf")
    other, venv2 = make_double(tmp_path / "b", n_steps=3)

    train_dqn.load_replay_buffer(other.replay_buffer, tmp_path / "buf")

    assert (tmp_path / "buf" / "next_observations.npy").exists()
    assert np.array_equal(other.replay_buffer.next_observations[:20], model.replay_buffer.next_observations[:20])


def test_メモリ節約モードの保存済みバッファの遷移を_新しい設定のバッファに取り込める(tmp_path):
    old, venv = make_double(tmp_path / "old", n_steps=1, double=False)
    train_dqn.train(old, tmp_path / "old", total_steps=100, checkpoint_every=1000)
    train_dqn.save_replay_buffer(old.replay_buffer, tmp_path / "buf")
    new, venv2 = make_double(tmp_path / "new", n_steps=3)

    count = train_dqn.import_transitions(new.replay_buffer, tmp_path / "buf")

    assert count > 0 and new.replay_buffer.size() == count
    old_obs, new_obs = old.replay_buffer.observations, new.replay_buffer.observations
    dones = old.replay_buffer.dones.reshape(-1)
    for i in range(count - 1):
        assert np.array_equal(new_obs[i], old_obs[i])
        if not dones[i]:                                      # 終了していなければ、次の観測は、次の添字の観測
            assert np.array_equal(new.replay_buffer.next_observations[i], old_obs[i + 1])
    assert np.array_equal(new.replay_buffer.rewards[:count], old.replay_buffer.rewards[:count])
    assert np.array_equal(new.replay_buffer.actions[:count], old.replay_buffer.actions[:count])


def test_末尾が終了していない遷移は_次の観測が無いので捨てる(tmp_path):
    old, venv = make_double(tmp_path / "old", n_steps=1, double=False)
    fill(old, 10)                                              # done が 1 件も無い
    train_dqn.save_replay_buffer(old.replay_buffer, tmp_path / "buf")
    new, venv2 = make_double(tmp_path / "new", n_steps=3)

    assert train_dqn.import_transitions(new.replay_buffer, tmp_path / "buf") == 9


def test_事前学習は_更新を行い_目標ネットワークを同期する(tmp_path):
    model, venv = make_double(tmp_path, n_steps=3)
    fill(model, 80)
    before = [p.clone() for p in model.q_net.parameters()]

    train_dqn.pretrain(model, updates=30, batch_size=8, chunk=10)

    assert model._n_updates == 30
    assert any(not th.equal(a, b) for a, b in zip(before, model.q_net.parameters()))
    assert all(th.equal(a, b) for a, b in zip(model.q_net.parameters(), model.q_net_target.parameters()))


def args(**overrides):
    base = dict(resume=None, init_buffer=None, pretrain_updates=0, double=False, n_steps=1)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_再開と新規のオプションの組み合わせを検査する(tmp_path):
    (tmp_path / "meta.json").write_text("{}")

    assert train_dqn.check_new_run_options(args()) is None
    assert train_dqn.check_new_run_options(args(double=True, n_steps=10, init_buffer=tmp_path, pretrain_updates=100)) is None
    assert "新しく始めるときだけ" in train_dqn.check_new_run_options(args(resume=tmp_path / "x.zip", double=True))
    assert "--init_buffer が必要" in train_dqn.check_new_run_options(args(pretrain_updates=100))
    assert "見つかりません" in train_dqn.check_new_run_options(args(init_buffer=tmp_path / "none"))


def test_探索の行動空間は_指定した割合と続ける確率で引く():
    from tools.exploration import PriorDiscrete
    prior = np.zeros(18)
    prior[9], prior[0] = 0.7, 0.3

    free = PriorDiscrete(prior, sticky=0.0, seed=0)
    sticky = PriorDiscrete(prior, sticky=0.9, seed=0)
    drawn = np.array([free.sample() for _ in range(4000)])
    held = [int(sticky.sample()) for _ in range(4000)]

    assert set(drawn.tolist()) == {0, 9} and (drawn == 9).mean() == pytest.approx(0.7, abs=0.03)
    assert sum(a != b for a, b in zip(held, held[1:])) / 3999 < 0.15         # 0.1 × (別の行動になる確率 0.42)
    assert isinstance(free.sample(), np.int64) and free.n == 18


def test_探索の行動空間は_不正な割合を受け付けない():
    from tools.exploration import PriorDiscrete
    with pytest.raises(ValueError):
        PriorDiscrete([0.0, 0.0])
    with pytest.raises(ValueError):
        PriorDiscrete([1.0, -0.5])
    with pytest.raises(ValueError):
        PriorDiscrete([1.0, 1.0], sticky=1.5)


def test_探索のランダムな行動が_人の割合から選ばれ_保存と再開をまたいで差し替えを戻せる(tmp_path):
    from tools.exploration import PriorDiscrete, set_exploration
    prior = np.zeros(18)
    prior[9] = 1.0                                              # 人は、常に行動 9 (止まる) を押す、という極端な割合
    model, venv = make_double(tmp_path, n_steps=3)
    model.exploration_initial_eps = 1.0
    set_exploration(model, PriorDiscrete(prior, sticky=0.5, seed=0))

    train_dqn.train(model, tmp_path, total_steps=100, checkpoint_every=1000)

    stored = model.replay_buffer.actions[:model.replay_buffer.size()].reshape(-1)
    # 探索率が高い最初の 10 ステップは、人の割合 (行動 9 だけ) から選ばれる。探索率が下がった後は、方策 (貪欲な行動) になる
    assert (stored[:10] == 9).mean() >= 0.9
    venv2 = train_dqn.build_vec_env(lambda: FakeGameEnv(), tmp_path, n_stack=4)
    resumed = train_dqn.load_for_resume(tmp_path / "final.zip", venv2, tmp_path, device="cpu")
    assert isinstance(resumed.action_space, PriorDiscrete)      # 差し替えは、保存に含まれる
    set_exploration(resumed, None)
    assert not isinstance(resumed.action_space, PriorDiscrete) and resumed.action_space.n == 18


def real_small_frame(name):
    """実機の撮影 (640x480) の自機エリアを、環境の観測と同じ 0.25 倍 (112x96) にしたもの"""
    import cv2
    from pathlib import Path
    from environment.screens import PLAYFIELD_RECT
    from utils import cv2helper
    gray = cv2helper.imread(str(Path(__file__).parent / "fixtures" / "screens" / f"{name}.png"), 0)
    x0, y0, x1, y1 = PLAYFIELD_RECT
    return cv2.resize(gray[y0:y1, x0:x1], (96, 112), interpolation=cv2.INTER_AREA)


def stack_of(frame):
    return np.broadcast_to(frame, (1, 4) + frame.shape).copy()


def chain(model, frames, last_done=True):
    """frames を順に、遷移 (frames[i] → frames[i+1]) としてバッファに追加する。報酬は 0、最後の遷移だけ done"""
    for i in range(len(frames) - 1):
        done = last_done and i == len(frames) - 2
        model.replay_buffer.add(stack_of(frames[i]), stack_of(frames[i + 1]), np.array([[i % 18]]), np.array([0.0]),
                                np.array([done]), [{}])


def test_見落とした死亡を直して取り込む_Restartの直前を死亡にし_Restartの観測から始まる遷移は捨てる(tmp_path):
    play, boss, restart = real_small_frame("playing"), real_small_frame("boss_battle"), real_small_frame("restart")
    frames = [play, boss, play, boss, play] + [restart] * 3 + [boss, play, boss, play, boss]
    old, venv = make_double(tmp_path / "old", n_steps=1, double=False)
    chain(old, frames)
    train_dqn.save_replay_buffer(old.replay_buffer, tmp_path / "buf")
    new, venv2 = make_double(tmp_path / "new", n_steps=3)

    count = train_dqn.import_transitions(new.replay_buffer, tmp_path / "buf")

    # 12 遷移のうち、Restart の観測から始まる 3 件を捨てる。play → Restart の遷移は、死亡 (done、報酬 -1) に直す
    assert count == 9 and new.replay_buffer.size() == 9
    dones = new.replay_buffer.dones[:9].reshape(-1)
    rewards = new.replay_buffer.rewards[:9].reshape(-1)
    assert dones.tolist() == [False, False, False, False, True, False, False, False, True]      # 4 番目 (直した死亡) と、最後 (元の done)
    assert rewards[4] == pytest.approx(-1.0) and rewards[[0, 1, 2, 3, 5, 6, 7, 8]].tolist() == [0.0] * 8
    assert np.array_equal(new.replay_buffer.next_observations[4][0, -1], restart)               # 死亡の遷移の次の観測は Restart


def test_repairを切ると_そのまま取り込む(tmp_path):
    play, restart = real_small_frame("playing"), real_small_frame("restart")
    old, venv = make_double(tmp_path / "old", n_steps=1, double=False)
    chain(old, [play, play, restart, restart, play])
    train_dqn.save_replay_buffer(old.replay_buffer, tmp_path / "buf")
    new, venv2 = make_double(tmp_path / "new", n_steps=3)

    count = train_dqn.import_transitions(new.replay_buffer, tmp_path / "buf", repair=False)

    assert count == 4 and not new.replay_buffer.dones[:4].reshape(-1)[:3].any()


def test_次の観測を持つバッファ_n_step_からも取り込め_見落としを直せる(tmp_path):
    play, restart = real_small_frame("playing"), real_small_frame("restart")
    old, venv = make_double(tmp_path / "old", n_steps=3)               # next_observations.npy を持つ
    chain(old, [play, play, restart, restart, restart, play, play])
    train_dqn.save_replay_buffer(old.replay_buffer, tmp_path / "buf")
    assert (tmp_path / "buf" / "next_observations.npy").exists()
    new, venv2 = make_double(tmp_path / "new", n_steps=3)

    count = train_dqn.import_transitions(new.replay_buffer, tmp_path / "buf")

    # 6 遷移: play→play、play→Restart (直して死亡)、Restart から始まる 3 件は捨てる、最後の play→play
    assert count == 3
    assert new.replay_buffer.dones[:3].reshape(-1).tolist() == [False, True, True]       # 2 番目: 直した死亡、3 番目: 元の done
