import numpy as np
import pytest

from tests.test_train_dqn import FakeGameEnv, PARAMS
from tools import evaluate_dqn, train_dqn


def make_venv(tmp_path, on_step=None):
    return train_dqn.build_vec_env(lambda: FakeGameEnv(on_step), tmp_path, n_stack=4)


def test_ランダム方策を指定した数のエピソード動かして_ステップ数と報酬を集計する(tmp_path, capsys):
    venv = make_venv(tmp_path)
    rng = np.random.default_rng(0)

    report = evaluate_dqn.evaluate(venv, 3, lambda obs: int(rng.integers(18)))

    assert len(report["episodes"]) == 3
    assert report["steps"]["median"] == FakeGameEnv.EPISODE_STEPS
    assert [e["reward"] for e in report["episodes"]] == [-100.0] * 3        # 記録は、縮める前の報酬
    assert report["interrupted"] is False and report["seconds"]["mean"] >= 0
    assert sum(share for _, share in report["top_actions"]) <= 1.0


def test_学習したモデルの方策で動かせて_選んだ行動の偏りが分かる(tmp_path, capsys):
    venv = make_venv(tmp_path)
    model = train_dqn.build_model(venv, tmp_path, PARAMS)
    eval_env = model.get_env()                     # モデルの形 (チャンネルが先) の観測を返す環境

    report = evaluate_dqn.evaluate(eval_env, 2, lambda obs: int(model.predict(obs, deterministic=True)[0][0]))

    assert len(report["episodes"]) == 2
    assert 0 < report["top_actions"][0][1] <= 1.0


def test_Ctrl_Cで中断すると_そこまでの結果を返す(tmp_path, capsys):
    def on_step(n: int) -> None:
        if n == FakeGameEnv.EPISODE_STEPS * 2 + 5:
            raise KeyboardInterrupt

    venv = make_venv(tmp_path, on_step)

    report = evaluate_dqn.evaluate(venv, 10, lambda obs: 0)

    assert report["interrupted"] is True
    assert len(report["episodes"]) == 2


def test_イベントはエピソードごとに集計される(tmp_path, capsys):
    class EventEnv(FakeGameEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            if self._steps == 10:
                info["events"] = ["chapter_finish"]
            return obs, reward, terminated, truncated, info

    venv = train_dqn.build_vec_env(lambda: EventEnv(), tmp_path, n_stack=4)

    report = evaluate_dqn.evaluate(venv, 2, lambda obs: 0)

    assert report["event_counts"] == {"chapter_finish": 2}
    assert [e["events"] for e in report["episodes"]] == [["chapter_finish"]] * 2
