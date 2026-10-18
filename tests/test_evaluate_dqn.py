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


def test_秒数には_エピソード終了時の自動リセットの復旧時間を含めない(tmp_path, capsys):
    state = {"now": 0.0}

    class SlowResetEnv(FakeGameEnv):
        def reset(self, *, seed=None, options=None):
            state["now"] += 5.0                      # 遷移制御による復旧の時間
            return super().reset(seed=seed, options=options)

        def step(self, action):
            state["now"] += 0.1                      # 1 ステップの時間
            return super().step(action)

    venv = train_dqn.build_vec_env(lambda: SlowResetEnv(), tmp_path, n_stack=4)

    report = evaluate_dqn.evaluate(venv, 2, lambda obs: 0, clock=lambda: state["now"])

    assert [e["seconds"] for e in report["episodes"]] == [pytest.approx(3.0)] * 2     # 30 ステップ × 0.1 秒


def test_人のプレイの行動の割合を求める_ボムを押したフレームとプレイ中でないフレームは除く(tmp_path):
    from environment.demo import DemoWriter
    from tools.evaluate_dqn import action_prior
    writer = DemoWriter(tmp_path, flush_every=50, background=False)
    screens = ["title"] * 3 + ["playing"] * 8 + ["game_over"]
    actions = [5] * 3 + [9, 9, 9, 0, 0, 12, 12, 12] + [3]
    for k, (screen, action) in enumerate(zip(screens, actions)):
        writer.add(np.zeros((8, 6), np.uint8), k * 0.133, screen, 0.0, False, 0, 0.5)
        writer.set_action(action, 1.0, k == 10, False)                # 10 番目 (行動 12) だけ、X を押した
    writer.close()

    prior = action_prior([tmp_path])

    assert prior.sum() == pytest.approx(1.0) and len(prior) == 18
    assert prior[9] == pytest.approx(3 / 7) and prior[0] == pytest.approx(2 / 7) and prior[12] == pytest.approx(2 / 7)
    assert prior[5] == 0 and prior[3] == 0


def test_割合から選ぶ方策は_stickyの確率で前の行動を続ける():
    from tools.evaluate_dqn import sampling_policy
    prior = np.full(18, 1 / 18)

    never = sampling_policy(prior, np.random.default_rng(0), sticky=0.0)
    always = sampling_policy(prior, np.random.default_rng(0), sticky=1.0)
    mostly = sampling_policy(prior, np.random.default_rng(0), sticky=0.9)

    def changes(policy):
        actions = [policy(None) for _ in range(2000)]
        return sum(a != b for a, b in zip(actions, actions[1:])) / 1999

    assert changes(always) == 0.0                                    # 最初に選んだ行動を、ずっと続ける
    assert changes(never) > 0.9
    assert 0.05 < changes(mostly) < 0.15                              # 続けない確率 0.1 × (別の行動になる確率 17/18)


def test_割合どおりに選ぶと_選んだ行動の割合が近づく():
    from tools.evaluate_dqn import sampling_policy
    prior = np.zeros(18)
    prior[9], prior[0] = 0.7, 0.3
    policy = sampling_policy(prior, np.random.default_rng(1))

    chosen = np.bincount([policy(None) for _ in range(4000)], minlength=18) / 4000

    assert chosen[9] == pytest.approx(0.7, abs=0.03) and chosen[0] == pytest.approx(0.3, abs=0.03)
    assert chosen.sum() == pytest.approx(chosen[9] + chosen[0])


def test_死なないまま最大ステップに達したエピソードは_打ち切りとして記録して_次に進む(tmp_path, capsys):
    class LongEnv(FakeGameEnv):
        EPISODE_STEPS = 50                                          # 最大 20 ステップより長い

    venv = train_dqn.build_vec_env(lambda: LongEnv(), tmp_path, n_stack=4)

    report = evaluate_dqn.evaluate(venv, 4, lambda obs: 0, max_steps=20)

    episodes = report["episodes"]
    assert [e["censored"] for e in episodes] == [True, True, False, True]    # 20 + 20 + 10 (50 ステップ目で死亡) + 次の 20
    assert [e["steps"] for e in episodes[:2]] == [20, 20] and report["censored"] == 3
    assert episodes[2]["steps"] == 50                                # 死亡したエピソードは、全体のステップ数


def test_最大ステップを指定しなければ_打ち切らない(tmp_path, capsys):
    venv = make_venv(tmp_path)

    report = evaluate_dqn.evaluate(venv, 2, lambda obs: 0)

    assert report["censored"] == 0 and all(not e["censored"] for e in report["episodes"])


def test_止まるを基本にした選び方は_余裕を超えて良い行動があるときだけ動く():
    from tools.evaluate_dqn import choose_with_margin
    q = np.zeros(18)
    q[9], q[3] = 0.50, 0.58                                      # 行動 3 は、止まる (9) より 0.08 良い

    assert choose_with_margin(q, 9, margin=0.05) == 3            # 余裕 0.05 を超える → 動く
    assert choose_with_margin(q, 9, margin=0.10) == 9            # 余裕 0.10 を超えない → 止まる
    assert choose_with_margin(q, 9, margin=0.0) == 3
    q[3] = q[9]
    assert choose_with_margin(q, 9, margin=0.0) == 9              # 同じ値なら、止まる


def test_Q値の最大は_モデルの貪欲な行動と一致する(tmp_path):
    from tools.evaluate_dqn import q_values
    venv = make_venv(tmp_path)
    model = train_dqn.build_model(venv, tmp_path, PARAMS)
    eval_env = model.get_env()
    obs = eval_env.reset()

    q = q_values(model, obs)

    assert q.shape == (18,)
    assert int(np.argmax(q)) == int(model.predict(obs, deterministic=True)[0][0])


def test_on_stepは_死なずに続いているステップだけ_何ステップ目かを渡して呼ばれる(tmp_path, capsys):
    venv = make_venv(tmp_path)                                       # 30 ステップで死ぬ
    seen = []

    evaluate_dqn.evaluate(venv, 2, lambda obs: 0, on_step=seen.append)

    assert seen == list(range(1, 30)) * 2                            # 死亡のステップ (30) では呼ばれない


def test_診断のためのenv_methodが_ラッパー越しに呼べる(tmp_path):
    class DumpEnv(FakeGameEnv):
        dumped = []

        def dump_diagnostics(self, reason):
            DumpEnv.dumped.append(reason)

    venv = train_dqn.build_vec_env(lambda: DumpEnv(), tmp_path, n_stack=4)

    venv.env_method("dump_diagnostics", "eval_step100")

    assert DumpEnv.dumped == ["eval_step100"]
