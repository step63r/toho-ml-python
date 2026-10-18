from pathlib import Path
from typing import Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import pytest

from gymnasium import spaces

from environment.wrappers import ChannelLast, ScaledReward
from tools import train_dqn

SHAPE = (112, 96)


class FakeGameEnv(gym.Env):
    """
    ゲームを起動せずに学習の流れを確かめるための環境

    EPISODE_STEPS ステップごとに死ぬ (reward=-100, terminated)。on_step で、指定したステップ目に副作用 (STOP ファイルの作成・例外) を起こせる。
    """
    EPISODE_STEPS = 30

    def __init__(self, on_step: Optional[Callable[[int], None]] = None) -> None:
        self.observation_space = spaces.Box(0, 255, SHAPE, np.uint8)
        self.action_space = spaces.Discrete(18)
        self.on_step = on_step
        self.total_steps = 0
        self._steps = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        return self._frame(), {}

    def _frame(self) -> np.ndarray:
        return np.full(SHAPE, self._steps % 256, dtype=np.uint8)

    def step(self, action):
        self.total_steps += 1
        self._steps += 1
        if self.on_step:
            self.on_step(self.total_steps)
        died = self._steps >= self.EPISODE_STEPS
        return self._frame(), (-100.0 if died else 0.0), died, False, {}

    def close(self) -> None:
        self.closed = True


PARAMS: Dict = dict(learning_rate=1e-4, buffer_size=300, learning_starts=40, batch_size=16, gamma=0.99, train_freq=4,
                    target_update_interval=100, exploration_fraction=0.3, exploration_final_eps=0.05, device="cpu",
                    seed=0)


def make_model(tmp_path: Path, on_step: Optional[Callable[[int], None]] = None):
    venv = train_dqn.build_vec_env(lambda: FakeGameEnv(on_step), tmp_path, n_stack=4, reward_scale=0.01)
    return train_dqn.build_model(venv, tmp_path, PARAMS), venv


def test_観測はチャンネル最後で4枚重ねになり_報酬は縮められ_Monitorには元の報酬が残る(tmp_path):
    venv = train_dqn.build_vec_env(lambda: FakeGameEnv(), tmp_path, n_stack=4, reward_scale=0.01)

    obs = venv.reset()
    rewards = []
    infos = []
    for _ in range(FakeGameEnv.EPISODE_STEPS):
        obs, reward, done, info = venv.step(np.array([0]))
        rewards.append(float(reward[0]))
        infos.append(info[0])

    assert venv.observation_space.shape == SHAPE + (4,)
    assert obs.shape == (1,) + SHAPE + (4,)
    assert rewards[-1] == pytest.approx(-1.0)                 # -100 を 0.01 倍
    assert infos[-1]["episode"]["r"] == pytest.approx(-100.0)  # 記録は縮める前
    assert infos[-1]["episode"]["l"] == FakeGameEnv.EPISODE_STEPS


def test_ラッパー単体(tmp_path):
    env = ScaledReward(ChannelLast(FakeGameEnv()), scale=0.5)

    obs, _ = env.reset()
    assert obs.shape == SHAPE + (1,) and env.observation_space.shape == SHAPE + (1,)
    assert env.step(0)[1] == 0.0


def test_目標まで学習すると_定期保存と最終保存が残る(tmp_path):
    model, venv = make_model(tmp_path)

    result = train_dqn.train(model, tmp_path, total_steps=300, checkpoint_every=100)

    assert result == "final"
    assert model.num_timesteps >= 300
    assert (tmp_path / "final.zip").exists() and (tmp_path / "final_replay_buffer.pkl").exists()
    assert sorted(p.name for p in (tmp_path / "checkpoints").glob("dqn_*_steps.zip")) == [
        "dqn_100_steps.zip", "dqn_200_steps.zip", "dqn_300_steps.zip"]
    assert not list(tmp_path.glob("*.partial.*"))             # 書き込み用の一時ファイルが残らない


def test_STOPファイルで止めると_保存して中断として終わる(tmp_path):
    def on_step(n: int) -> None:
        if n == 120:
            (tmp_path / train_dqn.STOP_FILE_NAME).write_text("")

    model, venv = make_model(tmp_path, on_step)

    result = train_dqn.train(model, tmp_path, total_steps=1000, checkpoint_every=500)

    assert result == "interrupted"
    assert 120 <= model.num_timesteps < 200
    assert (tmp_path / "interrupted.zip").exists() and (tmp_path / "interrupted_replay_buffer.pkl").exists()


def test_前回のSTOPファイルが残っていても_始めた直後に止まらない(tmp_path):
    (tmp_path / train_dqn.STOP_FILE_NAME).write_text("")
    model, venv = make_model(tmp_path)

    result = train_dqn.train(model, tmp_path, total_steps=100, checkpoint_every=500)

    assert result == "final"


def test_Ctrl_Cで止めると_保存して中断として終わる(tmp_path):
    def on_step(n: int) -> None:
        if n == 120:
            raise KeyboardInterrupt

    model, venv = make_model(tmp_path, on_step)

    result = train_dqn.train(model, tmp_path, total_steps=1000, checkpoint_every=500)

    assert result == "interrupted"
    assert (tmp_path / "interrupted.zip").exists()


def test_例外で止まっても_保存してから例外を送出する(tmp_path):
    def on_step(n: int) -> None:
        if n == 120:
            raise RuntimeError("ゲームが落ちた")

    model, venv = make_model(tmp_path, on_step)

    with pytest.raises(RuntimeError, match="ゲームが落ちた"):
        train_dqn.train(model, tmp_path, total_steps=1000, checkpoint_every=500)

    assert (tmp_path / "interrupted.zip").exists()


def test_中断したモデルから再開すると_ステップ数とバッファを引き継いで続きを学習する(tmp_path):
    def on_step(n: int) -> None:
        if n == 150:
            raise KeyboardInterrupt

    model, venv = make_model(tmp_path, on_step)
    assert train_dqn.train(model, tmp_path, total_steps=400, checkpoint_every=500) == "interrupted"
    stopped_at, buffered = model.num_timesteps, model.replay_buffer.size()
    venv.close()

    venv2 = train_dqn.build_vec_env(lambda: FakeGameEnv(), tmp_path, n_stack=4)
    resumed = train_dqn.load_for_resume(tmp_path / "interrupted.zip", venv2, tmp_path, device="cpu")

    assert resumed.num_timesteps == stopped_at
    assert resumed.replay_buffer.size() == buffered
    assert train_dqn.train(resumed, tmp_path, total_steps=400, checkpoint_every=500) == "final"
    assert resumed.num_timesteps >= 400
    # 再開しても、目標は累計のステップ数のまま (探索率の計画が、再開で伸びない)
    assert resumed._total_timesteps == 400


def test_リプレイバッファが無い再開では_貯まるまで学習を始めない(tmp_path):
    model, venv = make_model(tmp_path)
    train_dqn.train(model, tmp_path, total_steps=100, checkpoint_every=500)
    (tmp_path / "final_replay_buffer.pkl").unlink()
    venv2 = train_dqn.build_vec_env(lambda: FakeGameEnv(), tmp_path, n_stack=4)

    resumed = train_dqn.load_for_resume(tmp_path / "final.zip", venv2, tmp_path, device="cpu")

    assert resumed.learning_starts == resumed.num_timesteps + train_dqn.RESUME_WARMUP_STEPS
    assert resumed.replay_buffer.size() == 0


def test_環境を閉じるとキーが離される(tmp_path):
    env = FakeGameEnv()
    venv = train_dqn.build_vec_env(lambda: env, tmp_path, n_stack=4)

    venv.close()

    assert env.closed


def test_ログはファイルに追記され_ターミナルにはエラーだけが出る(tmp_path, capsys):
    import logging
    target = logging.getLogger("test_train_log")
    target.propagate = False

    path = train_dqn.setup_logging(tmp_path, root=target)
    target.info("学習を始めます")
    target.error("異常です")
    for handler in target.handlers:
        handler.flush()
    train_dqn.setup_logging(tmp_path, root=target)          # 再開しても、追記する
    target.info("再開しました")
    for handler in target.handlers:
        handler.flush()

    text = path.read_text(encoding="utf-8")
    assert path == tmp_path / train_dqn.LOG_FILE_NAME
    assert text.count("学習を始めます") == 1 and "異常です" in text and "再開しました" in text
    err = capsys.readouterr().err
    assert "異常です" in err and "学習を始めます" not in err


def test_進捗の表は_複数行のまま1件のログとして出る(tmp_path):
    import logging
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    target = logging.getLogger("test_log_stream")
    target.propagate = False
    target.setLevel(logging.INFO)
    target.addHandler(Collect())
    stream = train_dqn.LogStream(target)

    stream.write("-----\n| a | 1 |\n")
    stream.write("-----\n")
    stream.flush()
    stream.flush()                                           # 空の flush は出さない

    assert records == ["\n-----\n| a | 1 |\n-----"]


def test_学習の進捗は_ターミナルではなくログファイルとprogress_csvに出る(tmp_path, capsys):
    import logging
    sb3_log = logging.getLogger("sb3")
    path = train_dqn.setup_logging(tmp_path, root=sb3_log)
    sb3_log.propagate = False
    try:
        model, venv = make_model(tmp_path)
        capsys.readouterr()                                   # 構築時の出力を捨てる
        train_dqn.train(model, tmp_path, total_steps=300, checkpoint_every=1000)
        for handler in sb3_log.handlers:
            handler.flush()
    finally:
        sb3_log.handlers.clear()

    assert "ep_len_mean" in path.read_text(encoding="utf-8")
    out = capsys.readouterr()
    assert "ep_len_mean" not in out.out and "ep_len_mean" not in out.err
    assert list((tmp_path / "tb").glob("run_*/progress.csv"))
