from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

from typesafe_sdk import TypeSafeError

from tools import jev_probe

FIXTURES = Path(__file__).parent / "fixtures"


class FakeClient:
    """Jev の代わりに、決まった答えを返す。外部には一切通信しない"""
    def __init__(self, choice: str = "right", error: bool = False) -> None:
        self.choice = choice
        self.error = error
        self.calls = []
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(models=[
            SimpleNamespace(name="jev-test", release_date="2026-01-01", description="テスト用")]))

    def system_one(self, state: Any, questions: Dict[str, Any], **kwargs: Any) -> Any:
        self.calls.append((state, questions, kwargs))
        if self.error:
            raise TypeSafeError("boom")
        box = SimpleNamespace(top=0.7, bottom=0.8, left=0.4, right=0.5)
        return SimpleNamespace(
            choices={"move": SimpleNamespace(choice=self.choice,
                                             probabilities={"left": 0.1, "stay": 0.1, "right": 0.8})},
            nouls={"hit_soon": SimpleNamespace(noul=0.9)},
            boxes={"player": SimpleNamespace(box=box, confidence=0.9),
                   "enemy": SimpleNamespace(box=box, confidence=0.5)},
            usage=SimpleNamespace(input_tokens=100, output_tokens=0))


def test_場面は答えが明らかなものだけで_左右対称():
    cases = {c.name: c for c in jev_probe.build_cases()}

    assert len(cases) == 8
    assert all(c.acceptable <= {"left", "stay", "right"} and c.acceptable for c in cases.values())
    # 左が塞がれた場面と右が塞がれた場面は鏡像で、正解も逆になる
    assert cases["真上に弾・左が塞がれ右が空き"].acceptable == {"right"}
    assert cases["真上に弾・右が塞がれ左が空き"].acceptable == {"left"}
    left, right = (cases[n].state["bullets"] for n in ("真上に弾・左が塞がれ右が空き", "真上に弾・右が塞がれ左が空き"))
    assert sorted(round(1 - b["x"], 2) for b in left[1:]) == sorted(round(b["x"], 2) for b in right[1:])


def test_百分位と応答時間の要約():
    times = [0.1, 0.2, 0.3, 0.4, 1.0]

    assert jev_probe.percentile(times, 0.95) == 1.0
    assert jev_probe.summarize_latency(times) == {"min": 0.1, "p50": 0.3, "p95": 1.0, "max": 1.0}
    assert jev_probe.summarize_latency([]) == {}


@pytest.mark.parametrize("name", ["screens/stage2_play.png", "game_over_960x720.png"])
def test_ゲーム画面は自機エリアのRGB画像になる(name):
    # 保存済みのグレー全体画像 (640x480) もクライアント領域のRGB画像 (960x720) も同じ形に揃う
    assert jev_probe.load_game_frame(FIXTURES / name).shape == (450, 384, 3)


def test_座標は画像の割合として描き込まれる():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    jev_probe.draw_box(frame, SimpleNamespace(top=0.2, bottom=0.8, left=0.25, right=0.75), "x", (0, 255, 0))

    assert frame[20, 100, 1] == 255 and frame[50, 50, 1] == 255   # 上辺と左辺
    assert frame[50, 100, 1] == 0                                   # 内部は塗らない


def test_テキスト段階_採点と応答時間(capsys):
    client = FakeClient(choice="right")

    result = jev_probe.run_text(client, n_latency=3, model=None)

    # 「右」を許す場面は 3 つ (右が空き・両側が空き・左端)
    assert result["correct"] == 3 and result["answered"] == 8
    assert set(result["latency"]) == {"min", "p50", "p95", "max"}
    assert result["input_tokens"] == 100 * (8 + 3)
    assert len(client.calls) == 8 + 3


def test_失敗しても測定を続けてエラー内容を記録する():
    result = jev_probe.run_text(FakeClient(error=True), n_latency=2, model=None)

    assert result["answered"] == 0 and result["latency"] == {}
    assert all("error" in c for c in result["cases"])


def test_モデル一覧():
    assert jev_probe.run_models(FakeClient())["models"][0]["name"] == "jev-test"


def test_画像段階_送るのは指定した画像だけで_結果の画像を保存する(tmp_path):
    client = FakeClient()
    image = FIXTURES / "screens" / "stage2_play.png"

    result = jev_probe.run_images(client, [image], n_latency=2, model="jev-preview", out_dir=tmp_path)

    assert (tmp_path / "annotated_stage2_play.png").exists()
    assert result["images"][0]["player"]["confidence"] == 0.9
    assert len(client.calls) == 1 + 2
    # 画像は実験機能なので、オプトインのヘッダを必ず付ける (付けないと API は 400 を返す)
    assert all(call[2] == {"model": "jev-preview", "extra_headers": jev_probe.IMAGE_OPT_IN_HEADERS}
               for call in client.calls)
    assert jev_probe.IMAGE_OPT_IN_HEADERS == {"TypeSafe-Danger-Zone-Images": "true"}


def lanes_of(case_name: str):
    case = next(c for c in jev_probe.build_cases() if c.name == case_name)
    state = jev_probe.lane_state(case.state)
    return state["player_lane"], [lane["steps_until_hit"] for lane in state["lanes"]]


def test_レーン要約_弾が届くまでのステップ数と自機のレーン():
    assert lanes_of("弾なし") == (2, [None] * 5)
    assert lanes_of("弾は遠い") == (2, [None, None, 35.0, None, None])
    # 左が塞がれた場面: 自機のレーンと左のレーンが危険で、右のレーンが空いている
    assert lanes_of("真上に弾・左が塞がれ右が空き") == (2, [None, 3.0, 1.0, None, None])
    assert lanes_of("左端で真上に弾") == (0, [4.0, None, None, None, None])


def test_レーン要約_届かない弾は無視し_横に流れる弾は届く位置のレーンに数える():
    state = jev_probe.make_state(0.5, [
        (0.5, 0.95, 0, 0.02),      # すでに自機の高さを過ぎた
        (0.5, 0.5, 0, -0.02),      # 上へ向かう
        (0.1, 0.8, 0.03, 0.02),    # 5 ステップ後に x=0.25 → レーン1 にずれて届く (境界から離した値)
    ])

    lanes = jev_probe.lane_state(state)["lanes"]

    assert [lane["steps_until_hit"] for lane in lanes] == [None, 5.0, None, None, None]


def test_レーン段階_採点と応答時間():
    client = FakeClient(choice="right")

    result = jev_probe.run_lanes(client, n_latency=2, model=None)

    # 正解は機械的な最適手。「右」が正解なのは、遠い弾 (左右どちらも可)・左が塞がれ右が空き・両側が空き・左端の4つ
    assert result["correct"] == 4 and result["answered"] == 8
    # 送った状態は座標リストではなくレーン要約で、質問も専用のものになっている
    state, questions, _ = client.calls[0]
    assert "lanes" in state and "bullets" not in state
    assert questions is jev_probe.LANE_QUESTIONS


def lane_info_of(player_lane, steps):
    return {"player_lane": player_lane, "lanes": [{"lane": i, "steps_until_hit": v} for i, v in enumerate(steps)]}


def test_最適手_弾のないレーンが最も安全で_同点なら動かない():
    # 自機のレーン2が危険 (4ステップ)・左が3・右は空き → 右
    assert jev_probe.oracle_moves(lane_info_of(2, [None, 3.0, 4.0, None, None]))[0] == {"right"}
    # 両隣のほうが危険 → 動かない
    assert jev_probe.oracle_moves(lane_info_of(2, [None, 2.5, None, 2.5, None]))[0] == {"stay"}
    # 左右とも空きで自機のレーンが危険 → どちらでもよい (同点)
    assert jev_probe.oracle_moves(lane_info_of(2, [None, None, 4.0, None, None]))[0] == {"left", "right"}
    # 何も来ない場合は動かない
    assert jev_probe.oracle_moves(lane_info_of(2, [None] * 5))[0] == {"stay"}
    # 端では、無い側へは動けない
    assert jev_probe.oracle_moves(lane_info_of(0, [4.0, None, None, None, None]))[0] == {"right"}


def test_最適手_余白は最良と次点の差():
    assert jev_probe.oracle_moves(lane_info_of(2, [None, 3.0, 4.0, 9.0, None]))[1] == pytest.approx(5.0)
    assert jev_probe.oracle_moves(lane_info_of(2, [None, 3.0, 4.0, None, None]))[1] == float("inf")


def test_ランダム場面_決定的で_クラスが均等で_正解が一意で余白がある():
    scenarios = jev_probe.random_scenarios(5, seed=1)

    assert scenarios == jev_probe.random_scenarios(5, seed=1)
    assert len(scenarios) == 15
    assert sorted(truth for _, truth in scenarios) == ["left"] * 5 + ["right"] * 5 + ["stay"] * 5
    for state, truth in scenarios:
        moves, margin = jev_probe.oracle_moves(jev_probe.lane_state(state))
        assert moves == {truth} and margin >= 2.0


def test_正解を機械的に求める検証_偏った答えは一致率に出る(capsys):
    client = FakeClient(choice="left")

    report = jev_probe.run_oracle(client, per_class=4, model=None)

    # 常に「左」と答えると、左が正解の 1/3 だけ合う。混同行列に偏りが出る
    assert report["lanes"]["correct"] == 4 and report["lanes"]["total"] == 12
    assert report["lanes"]["confusion"]["left"] == {"left": 4, "stay": 0, "right": 0}
    assert report["lanes"]["confusion"]["right"] == {"left": 4, "stay": 0, "right": 0}
    assert report["raw"]["total"] == 12
    assert len(client.calls) == 24
