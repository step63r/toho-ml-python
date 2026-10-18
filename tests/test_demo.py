from typing import Any, Dict, List

import numpy as np
import pytest

from environment import demo
from environment.demo import (FLAG_CHAPTER_FINISH, FLAG_STAGE_CLEAR, SCREEN_NAMES, DemoWriter, KeySampler, action_to_keys,
                              find_episodes, keys_to_action, load_session, summarize_interval)
from tools.record_demo import record
from utils.native import COMMAND_DICT

NONE = (False,) * 8


def state(left=False, right=False, up=False, down=False, shift=False, z=True, x=False, esc=False):
    return (left, right, up, down, shift, z, x, esc)


def test_キーの組み合わせから環境の18通りの行動を求める_環境のCOMMAND_DICTと一致する():
    for action in range(18):
        left, right, up, down, shift = action_to_keys(action)
        assert keys_to_action(left, right, up, down, shift) == action
    assert len([a for a in COMMAND_DICT if a >= 0]) == 18


def test_左右や上下を同時に押したら打ち消し合う():
    assert keys_to_action(True, True, False, False, False) == 0
    assert keys_to_action(True, False, True, True, True) == 1 + 9          # 上下が打ち消し、左だけ + Shift


def test_区間のキーの状態から_最も長く押していた行動を選ぶ():
    samples = [(0.0, state(left=True)), (0.02, state(right=True)), (0.10, state())]

    result = summarize_interval(samples, 0.0, 0.133)

    assert result["action"] == 3                      # 右を 0.08 秒。左 0.02 秒・無し 0.033 秒
    assert result["z_fraction"] == pytest.approx(1.0)


def test_同じ長さなら後の行動を選ぶ():
    samples = [(0.0, state(left=True)), (0.05, state(right=True))]

    assert summarize_interval(samples, 0.0, 0.10)["action"] == 3


def test_Zの割合とXとEscを記録する():
    samples = [(0.0, state(z=True)), (0.05, state(z=False, x=True)), (0.1, state(z=False, esc=True))]

    result = summarize_interval(samples, 0.0, 0.2)

    assert result["z_fraction"] == pytest.approx(0.25)
    assert result["x_pressed"] is True and result["esc_pressed"] is True


def test_サンプルが無ければ行動は未確定_マイナス1():
    assert summarize_interval([], 0.0, 0.1)["action"] == -1


def test_KeySamplerは区間ごとに返し_押しっぱなしのキーを次の区間に引き継ぐ():
    now = {"t": 0.0}
    current = {"state": state(left=True)}
    sampler = KeySampler(read=lambda: current["state"], clock=lambda: now["t"])
    for t in (0.01, 0.05, 0.09):
        now["t"] = t
        sampler.sample()
    first = sampler.consume(0.0, 0.1)
    current["state"] = state(left=True)
    now["t"] = 0.15
    sampler.sample()
    second = sampler.consume(0.1, 0.2)

    assert [t for t, _ in first] == [0.01, 0.05, 0.09]
    assert second[0] == (0.1, state(left=True))                 # 区間の頭に、直前の状態 (左を押したまま) を引き継ぐ
    assert [t for t, _ in second][1:] == [0.15]
    assert summarize_interval(second, 0.1, 0.2)["action"] == 1


def test_DemoWriterは_行動が決まったフレームから書き_最後のフレームは行動なしで閉じる(tmp_path):
    writer = DemoWriter(tmp_path, flush_every=2, background=False, meta={"period": 0.133})
    frame = lambda v: np.full((4, 3), v, dtype=np.uint8)

    for k in range(5):
        writer.add(frame(k), t=k * 0.133, screen="playing", reward=float(k), terminated=False, flags=0, game_over_score=0.5)
        if k < 4:
            writer.set_action(action=k, z_fraction=1.0, x_pressed=False, esc_pressed=False)
    meta = writer.close()

    session = load_session(tmp_path, with_obs=True)
    # 行動が決まった 2 件ずつ書き (part_0000)、残りは close で書く (part_0001)
    assert meta["frames"] == 5 and sorted(p.name for p in tmp_path.glob("part_*.npz")) == ["part_0000.npz", "part_0001.npz"]
    assert session["action"].tolist() == [0, 1, 2, 3, -1]            # 最後のフレームの行動は未確定
    assert session["obs"].shape == (5, 4, 3) and session["obs"][:, 0, 0].tolist() == [0, 1, 2, 3, 4]
    assert session["reward"].tolist() == [0, 1, 2, 3, 4]
    assert not list(tmp_path.glob("*.partial.npz"))


def test_画像を読まずに_数値だけを読める(tmp_path):
    writer = DemoWriter(tmp_path, flush_every=10, background=False)
    writer.add(np.zeros((4, 3), np.uint8), 0.0, "playing", 0.0, False, 0, 0.0)
    writer.close()

    assert "obs" not in load_session(tmp_path)


def test_記録が無ければ例外(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path)


class FakeEnv:
    """observe() が、台本どおりの画面・報酬・出来事を返す環境"""
    def __init__(self, script: List[Dict[str, Any]]) -> None:
        self.script = script
        self.calls = 0

    def observe(self):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        info = {"events": step.get("events", []), "screen": step.get("screen", "playing"), "game_over_score": 0.5}
        return np.full((4, 3), self.calls, dtype=np.uint8), step.get("reward", 0.0), step.get("terminated", False), False, info


class ScriptedSampler:
    """consume で、フレームごとの台本のキー状態を返す"""
    def __init__(self, states) -> None:
        self.states = list(states)

    def consume(self, start, end):
        key = self.states.pop(0) if self.states else state()
        return [(start, key)]


class FakeClock:
    def __init__(self, step_cost: float = 0.0) -> None:
        self.now = 0.0
        self.step_cost = step_cost

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_記録は_周期ごとに撮って_前の周期のキーを行動にする(tmp_path):
    fake = FakeClock()
    script = [{"screen": "playing"}] * 3 + [{"screen": "game_over", "reward": -100.0, "terminated": True, "events": ["game_over"]}]
    env = FakeEnv(script)
    sampler = ScriptedSampler([state(left=True), state(up=True), state(right=True, shift=True)])
    writer = DemoWriter(tmp_path, background=False)
    frames = {"n": 0}

    def should_stop():
        frames["n"] += 1
        return frames["n"] > 4                                    # 4 フレーム撮って止める

    result = record(env, sampler, writer, period=0.1, should_stop=should_stop, clock=fake.clock, sleep=fake.sleep)
    writer.close()

    session = load_session(tmp_path)
    assert result["frames"] == 4 and result["deaths"] == 1 and result["late_frames"] == 0
    assert [SCREEN_NAMES[i] for i in session["screen"]] == ["playing"] * 3 + ["game_over"]
    # フレーム k の行動は、フレーム k と k+1 の間のキー。最後のフレームは、次の撮影が無く、区間の長さが 0 なので、未確定 (-1)
    assert session["action"].tolist() == [1, 2, 12, -1]
    assert session["terminated"].tolist() == [False, False, False, True]
    assert session["flags"][3] & demo.FLAG_GAME_OVER
    assert session["t"].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_遅れたフレームは数えて_詰めて撮らない(tmp_path):
    fake = FakeClock()
    env = FakeEnv([{"screen": "playing"}] * 6)
    original = env.observe

    def slow_observe():
        if env.calls == 2:
            fake.now += 0.35                                      # 撮影が長引いた
        return original()

    env.observe = slow_observe
    writer = DemoWriter(tmp_path, background=False)
    counter = {"n": 0}

    def should_stop():
        counter["n"] += 1
        return counter["n"] > 5

    result = record(env, ScriptedSampler([]), writer, period=0.1, should_stop=should_stop, clock=fake.clock, sleep=fake.sleep)

    assert result["late_frames"] == 1
    writer.close()
    ts = load_session(tmp_path)["t"]
    assert all(b - a >= 0.099 for a, b in zip(ts, ts[1:]))        # 遅れを取り戻すために、間隔を詰めない


def test_チャプターなどの出来事は_フラグと件数に残る(tmp_path):
    fake = FakeClock()
    env = FakeEnv([{"events": ["chapter_finish"], "reward": 100.0}, {}, {"events": ["stage_clear"], "reward": 1000.0, "terminated": True}])
    writer = DemoWriter(tmp_path, background=False)
    counter = {"n": 0}

    def should_stop():
        counter["n"] += 1
        return counter["n"] > 3

    result = record(env, ScriptedSampler([]), writer, period=0.1, should_stop=should_stop, clock=fake.clock, sleep=fake.sleep)
    writer.close()

    session = load_session(tmp_path)
    assert result["chapter_finish"] == 1 and result["stage_clear"] == 1
    assert session["flags"].tolist() == [FLAG_CHAPTER_FINISH, 0, FLAG_STAGE_CLEAR]


def test_最大時間で終わる(tmp_path):
    fake = FakeClock()
    writer = DemoWriter(tmp_path, background=False)

    result = record(FakeEnv([{}]), ScriptedSampler([]), writer, period=0.1, max_seconds=0.5, clock=fake.clock, sleep=fake.sleep)

    assert 5 <= result["frames"] <= 6


def make_session(screens, flags=None):
    n = len(screens)
    return {"screen": np.array([SCREEN_NAMES.index(s) for s in screens]),
            "flags": np.array(flags if flags is not None else [0] * n)}


def test_エピソードは_死亡で終わり_ゲームオーバーの間は含めず_リトライ後に新しく始まる():
    session = make_session(["loading", "playing"] * 1 + ["playing"] * 4 + ["game_over"] * 3 + ["unknown", "playing", "playing", "playing"])

    episodes = find_episodes(session)

    assert [(e["start"], e["end"], e["reason"]) for e in episodes] == [(1, 6, "death"), (10, 12, "open")]


def test_ポーズやメニューでプレイ画面でなくなったら_そこで終わる():
    session = make_session(["playing"] * 3 + ["manual_menu"] * 2 + ["playing"] * 2)

    episodes = find_episodes(session)

    assert [(e["start"], e["end"], e["reason"]) for e in episodes] == [(0, 3, "left"), (5, 6, "open")]


def test_ステージクリアで終わり_次のステージが始まるまで新しいエピソードを始めない():
    session = make_session(["playing"] * 3 + ["stage_clear"] * 3 + ["loading", "unknown", "playing", "playing"],
                           flags=[0, 0, 0, FLAG_STAGE_CLEAR, 0, 0, 0, 0, 0, 0])

    episodes = find_episodes(session)

    assert [(e["start"], e["end"], e["reason"]) for e in episodes] == [(0, 3, "clear"), (8, 9, "open")]


def write_session(tmp_path, screens, actions, flags=None, x=None, z=1.0, period=0.133):
    writer = DemoWriter(tmp_path, flush_every=50, background=False)
    for k, name in enumerate(screens):
        writer.add(np.full((8, 6), k, np.uint8), k * period, name, 0.0, False, 0 if flags is None else flags[k], 0.5)
        writer.set_action(actions[k], z, bool(x and x[k]), False)
    writer.close()


def test_確認ツールは_エピソードと押したキーの偏りと注意を出す(tmp_path):
    from tools import inspect_demo
    screens = ["playing"] * 10 + ["game_over"] * 3 + ["playing"] * 5
    actions = [3, 3, 3, 12, 12, 0, 0, 3, 3, 3] + [0] * 3 + [1] * 5
    write_session(tmp_path, screens, actions, x=[False] * 4 + [True] * 2 + [False] * 12, z=0.3)

    stats = inspect_demo.summarize(load_session(tmp_path), 0.133)
    messages = inspect_demo.warnings(stats, 0.133)

    assert [(e["start"], e["end"], e["reason"]) for e in stats["episodes"]] == [(0, 10, "death"), (13, 17, "open")]
    assert stats["playable_frames"] == 10 + 4                         # 終わりのフレーム (遷移の次の画面) は含めない
    assert stats["actions"][3] == 6 and stats["x_pressed_frames"] == 2
    assert stats["z_fraction"] == pytest.approx(0.3, abs=0.01)
    assert any("ボム" in m for m in messages) and any("Z (ショット)" in m for m in messages)


def test_確認ツールは_周期のずれと遅れを注意する(tmp_path):
    from tools import inspect_demo
    writer = DemoWriter(tmp_path, flush_every=50, background=False)
    t = 0.0
    for k in range(20):
        writer.add(np.zeros((8, 6), np.uint8), t, "playing", 0.0, False, 0, 0.5)
        writer.set_action(3, 1.0, False, False)
        t += 0.4 if k % 4 == 0 else 0.133
    writer.close()

    stats = inspect_demo.summarize(load_session(tmp_path), 0.133)

    assert stats["late_frames"] == 5
    assert any("遅れた" in m for m in inspect_demo.warnings(stats, 0.133))


def test_確認ツールは_問題の無い記録に注意を出さない(tmp_path):
    from tools import inspect_demo
    write_session(tmp_path, ["playing"] * 30, [3, 12, 5] * 10)

    stats = inspect_demo.summarize(load_session(tmp_path), 0.133)

    assert inspect_demo.warnings(stats, 0.133) == []


def test_画像の一覧を書き出せる(tmp_path):
    from tools import inspect_demo
    write_session(tmp_path / "s", ["playing"] * 30, [3, 12, 5] * 10)

    inspect_demo.sheet([tmp_path / "s"], tmp_path / "sheet.png")

    assert (tmp_path / "sheet.png").stat().st_size > 0


def run_unknown(tmp_path, seconds):
    fake = FakeClock()
    messages = []
    record(FakeEnv([{"screen": "unknown"}]), ScriptedSampler([]), DemoWriter(tmp_path, background=False), period=1.0,
           max_seconds=seconds, clock=fake.clock, sleep=fake.sleep, status=messages.append, status_every=1000.0)
    return [m for m in messages if "隠れていないか" in m]


def test_画面が判定できない状態が続いたら_ゲームが隠れていないか知らせる_短い間は知らせない(tmp_path):
    assert run_unknown(tmp_path / "short", 14.0) == []             # ローディングなどの短い unknown では警告しない
    assert len(run_unknown(tmp_path / "once", 40.0)) == 1          # 15 秒後に 1 回
    assert len(run_unknown(tmp_path / "twice", 70.0)) == 2         # その 30 秒後 (45 秒) にもう 1 回
