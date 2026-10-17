"""
Jev の疎通・応答時間・判断の質・画像対応を実測する

段階 (--stage):
  models : 使えるモデルの一覧
  text   : 合成した「自機と弾」の状態で、応答時間と、答えが明らかな場面での判断を測る (ゲーム画面は送らない)
  lanes  : 同じ場面を、5レーンごとの「弾が届くまでのステップ数」に要約して渡し、判断が改善するかを測る
  oracle : ランダムな場面で、機械的に計算した正解 (最も安全なレーン) との一致率を、座標とレーン要約の両形式で測る
  image  : --images で指定したゲーム画面を送り、自機・ボスの座標 (Box) と判断を測る

API キーは環境変数 TYPESAFE_API_KEY から SDK が読む。キーは表示も保存もしない。
結果は --out_dir (既定 jev_probe_out/) に results.json と、座標を描き込んだ画像で保存する。

    python -m tools.jev_probe --stage models text lanes
    python -m tools.jev_probe --stage image --images tests/fixtures/screens/stage2_play.png
"""
import argparse
import io
import json
import math
import random
import statistics
import time

import cv2
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from PIL import Image as PILImage
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from typesafe_sdk import Box, Choice, Image, Noul, RetryPolicy, TypeSafeClient, TypeSafeError

from environment.capture import FrameSpec, preprocess
from environment.screens import PLAYFIELD_RECT

# 公式の入力トークン単価 (USD / トークン)。出力は無料とされている
INPUT_USD_PER_TOKEN = 0.042 / 1_000_000

# 画像は実験機能で、このヘッダを付けないと 400 になる (API のエラー本文の指示)
IMAGE_OPT_IN_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}

LANES = 5

DIRECTION_QUESTIONS = {
    "move": Choice(
        instructions="Which way should the player move horizontally to avoid being hit by the bullets?",
        criteria={
            "left": "Move left (decrease x).",
            "stay": "Stay at the current x.",
            "right": "Move right (increase x).",
        }),
    "hit_soon": Noul(instructions="A bullet will hit the player within the next few steps if the player does not move."),
}

STATE_NOTE = ("Normalized coordinates: x from 0 (left) to 1 (right), y from 0 (top) to 1 (bottom). "
              "Each bullet moves by (vx, vy) per step. The player is at the bottom and can only move horizontally.")


@dataclass(frozen=True)
class Case:
    """答えが明らかな場面。acceptable に含まれる選択なら正解とする"""
    name: str
    state: Dict[str, Any]
    acceptable: FrozenSet[str]
    danger: bool


def make_state(player_x: float, bullets: Sequence[Tuple[float, float, float, float]]) -> Dict[str, Any]:
    return {
        "note": STATE_NOTE,
        "player": {"x": player_x, "y": 0.9},
        "bullets": [{"x": x, "y": y, "vx": vx, "vy": vy} for x, y, vx, vy in bullets],
    }


def build_cases() -> List[Case]:
    down = 0.02
    wall_left = [(0.30, 0.80, 0, down), (0.36, 0.84, 0, down), (0.42, 0.78, 0, down), (0.40, 0.88, 0, down)]
    wall_right = [(1 - x, y, vx, vy) for x, y, vx, vy in wall_left]
    return [
        Case("弾なし", make_state(0.5, []), frozenset({"stay"}), False),
        Case("弾は遠い", make_state(0.5, [(0.5, 0.2, 0, down)]), frozenset({"stay"}), False),
        Case("真上に弾・左が塞がれ右が空き", make_state(0.5, [(0.5, 0.80, 0, down)] + wall_left), frozenset({"right"}), True),
        Case("真上に弾・右が塞がれ左が空き", make_state(0.5, [(0.5, 0.80, 0, down)] + wall_right), frozenset({"left"}), True),
        Case("真上に弾・両側が空き", make_state(0.5, [(0.5, 0.82, 0, down)]), frozenset({"left", "right"}), True),
        Case("左端で真上に弾", make_state(0.03, [(0.03, 0.82, 0, down)]), frozenset({"right"}), True),
        Case("右端で真上に弾", make_state(0.97, [(0.97, 0.82, 0, down)]), frozenset({"left"}), True),
        Case("同じ高さの両脇を弾が通過", make_state(0.5, [(0.35, 0.85, 0, down), (0.65, 0.85, 0, down)]),
             frozenset({"stay"}), False),
    ]


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)]


def describe_error(error: BaseException) -> str:
    status = getattr(error, "status", None)
    request_id = getattr(error, "request_id", None)
    return f"{type(error).__name__} status={status} request_id={request_id}: {str(error)[:300]}"


def ask(client: Any, state: Any, questions: Dict[str, Any], **kwargs: Any) -> Tuple[Optional[Any], float, Optional[str]]:
    """(応答, 秒, エラー文)。失敗しても例外にせず、実測を続けられるようにする"""
    start = time.perf_counter()
    try:
        response = client.system_one(state=state, questions=questions, **kwargs)
        return response, time.perf_counter() - start, None
    except TypeSafeError as error:
        return None, time.perf_counter() - start, describe_error(error)


def run_models(client: Any) -> Dict[str, Any]:
    try:
        models = client.models.list().models
    except TypeSafeError as error:
        print(f"モデル一覧の取得に失敗: {describe_error(error)}")
        return {"error": describe_error(error)}
    for model in models:
        print(f"  {model.name:20s} {model.release_date}  {model.description}")
    return {"models": [{"name": m.name, "release_date": m.release_date, "description": m.description} for m in models]}


def lane_state(state: Dict[str, Any], lanes: int = LANES) -> Dict[str, Any]:
    """
    座標リストを、レーンごとの「最初の弾が自機の高さに届くまでのステップ数」に要約する

    弾は等速直線運動と見なし、届く時点の x が属するレーンに数える。届かない弾 (上へ向かう・既に通過) は無視する。
    """
    player_y = state["player"]["y"]
    arrival: Dict[int, float] = {}
    for bullet in state["bullets"]:
        if bullet["vy"] <= 0 or bullet["y"] >= player_y:
            continue
        steps = (player_y - bullet["y"]) / bullet["vy"]
        lane = min(max(int((bullet["x"] + bullet["vx"] * steps) * lanes), 0), lanes - 1)
        arrival[lane] = min(arrival.get(lane, steps), steps)
    return {
        "note": ("The field is split into lanes numbered from 0 (left) to {} (right). "
                 "steps_until_hit is when the first bullet reaches the player's height in that lane "
                 "(smaller is more dangerous, null means nothing is coming).").format(lanes - 1),
        "player_lane": min(int(state["player"]["x"] * lanes), lanes - 1),
        "lanes": [{"lane": i, "steps_until_hit": None if i not in arrival else round(arrival[i], 1)} for i in range(lanes)],
    }


LANE_QUESTIONS = {
    "move": Choice(
        instructions="Choose the move that puts the player in the safest lane. A lane with a smaller steps_until_hit is more dangerous.",
        criteria={
            "left": "Move to the adjacent lane on the left.",
            "stay": "Stay in the current lane.",
            "right": "Move to the adjacent lane on the right.",
        }),
    "hit_soon": Noul(instructions="The player's current lane will be hit within a few steps."),
}


def oracle_moves(lane_info: Dict[str, Any]) -> Tuple[FrozenSet[str], float]:
    """
    「現在のレーンと隣接レーンのうち、弾が最も遅く届く (来ない) レーン」へ動く手を、機械的に求める

    同点なら、動かない (stay) を優先する。余白 (margin) は、最良と次点の差 (ステップ)。
    """
    player = lane_info["player_lane"]
    steps = {lane["lane"]: math.inf if lane["steps_until_hit"] is None else lane["steps_until_hit"]
             for lane in lane_info["lanes"]}
    values = {move: steps[player + offset] for move, offset in (("left", -1), ("stay", 0), ("right", 1))
              if player + offset in steps}
    best = max(values.values())
    others = [v for v in values.values() if v != best]
    margin = best - max(others) if others else math.inf
    winners = frozenset(m for m, v in values.items() if v == best)
    return (frozenset({"stay"}) if "stay" in winners else winners), margin


def random_scenarios(per_class: int, seed: int = 0, lanes: int = LANES) -> List[Tuple[Dict[str, Any], str]]:
    """
    ランダムな場面を、正解 (left / stay / right) ごとに同数集める

    正解が1つに決まり、最良と次点の差が2ステップ以上あり、周囲に危険がある場面だけを使う
    (差が小さい場面や、何も来ない場面では、正解の判定自体が曖昧になるため)。
    """
    rng = random.Random(seed)
    buckets: Dict[str, List[Tuple[Dict[str, Any], str]]] = {"left": [], "stay": [], "right": []}
    for _ in range(100_000):
        if all(len(b) >= per_class for b in buckets.values()):
            break
        player_lane = rng.randrange(lanes)
        player_x = (player_lane + 0.5) / lanes + rng.uniform(-0.05, 0.05)
        bullets = [(rng.random(), rng.uniform(0.3, 0.85), rng.uniform(-0.01, 0.01), rng.uniform(0.01, 0.03))
                   for _ in range(rng.randint(3, 10))]
        state = make_state(player_x, bullets)
        info = lane_state(state, lanes)
        moves, margin = oracle_moves(info)
        if len(moves) != 1 or margin < 2.0 or all(lane["steps_until_hit"] is None for lane in info["lanes"]):
            continue
        move = next(iter(moves))
        if len(buckets[move]) < per_class:
            buckets[move].append((state, move))
    return [item for bucket in buckets.values() for item in bucket]


# 正解を機械的に求める検証では、質問の文言を正解の定義と一致させる
ORACLE_RAW_NOTE = ("Normalized coordinates: x from 0 (left) to 1 (right), y from 0 (top) to 1 (bottom). Each bullet moves by "
                   "(vx, vy) per step. Split the field into 5 equal-width lanes (lane 0 is x in [0, 0.2), lane 4 is [0.8, 1.0]). "
                   "A bullet hits a lane when it reaches the player's y with its x in that lane. Choose the move that puts the "
                   "player in the safest lane among the current lane and its adjacent lanes: the safest lane is the one whose "
                   "first bullet arrives latest, and a lane with no bullet arriving is the safest.")


def run_oracle(client: Any, per_class: int, model: Optional[str], seed: int = 0) -> Dict[str, Any]:
    kwargs = {"model": model} if model else {}
    scenarios = random_scenarios(per_class, seed)
    report: Dict[str, Any] = {"scenarios": len(scenarios)}
    for name, state_of in (("lanes", lane_state), ("raw", lambda state: {**state, "note": ORACLE_RAW_NOTE})):
        confusion = {t: {p: 0 for p in ("left", "stay", "right")} for t in ("left", "stay", "right")}
        times, tokens, failed = [], 0, 0
        for state, truth in scenarios:
            response, seconds, error = ask(client, state_of(state), {"move": LANE_QUESTIONS["move"]}, **kwargs)
            if error:
                failed += 1
                continue
            confusion[truth][response.choices["move"].choice] += 1
            times.append(seconds)
            tokens += response.usage.input_tokens or 0
        total = sum(sum(row.values()) for row in confusion.values())
        correct = sum(confusion[m][m] for m in confusion)
        print(f"--- {name}: 正解 {correct}/{total} ({correct / max(total, 1):.0%}) 失敗 {failed}")
        for truth, row in confusion.items():
            n = sum(row.values())
            print(f"  正解={truth:5s} -> " + " ".join(f"{p}={c}" for p, c in row.items())
                  + (f"  (一致 {row[truth] / n:.0%})" if n else ""))
        if times:
            print(f"  応答時間 p50={statistics.median(times) * 1000:.0f}ms 入力トークン {tokens}")
        report[name] = {"confusion": confusion, "correct": correct, "total": total, "failed": failed,
                        "latency": summarize_latency(times), "input_tokens": tokens}
    return report


def run_text(client: Any, n_latency: int, model: Optional[str]) -> Dict[str, Any]:
    return run_cases(client, DIRECTION_QUESTIONS, lambda case: case.state, lambda case: case.acceptable,
                     n_latency, model)


def run_lanes(client: Any, n_latency: int, model: Optional[str]) -> Dict[str, Any]:
    # 質問が「最も安全なレーン」なので、正解は機械的な最適手 (弾のないレーンへ動くのも正解)
    return run_cases(client, LANE_QUESTIONS, lambda case: lane_state(case.state),
                     lambda case: oracle_moves(lane_state(case.state))[0], n_latency, model)


def run_cases(client: Any, questions: Dict[str, Any], state_of: Any, acceptable_of: Any, n_latency: int,
              model: Optional[str]) -> Dict[str, Any]:
    kwargs = {"model": model} if model else {}
    results, correct, tokens = [], 0, 0
    print("--- 判断の質 (答えが明らかな場面)")
    for case in build_cases():
        response, seconds, error = ask(client, state_of(case), questions, **kwargs)
        if error:
            print(f"  {case.name}: 失敗 {error}")
            results.append({"case": case.name, "error": error})
            continue
        move, hit = response.choices["move"], response.nouls["hit_soon"].noul
        ok = move.choice in acceptable_of(case)
        correct += ok
        tokens += response.usage.input_tokens or 0
        probs = " ".join(f"{k}={v:.2f}" for k, v in move.probabilities.items())
        print(f"  {'OK' if ok else 'NG'} {case.name}: {move.choice} ({probs}) hit_soon={hit:.2f} {seconds * 1000:.0f}ms")
        results.append({"case": case.name, "choice": move.choice, "probabilities": move.probabilities,
                        "hit_soon": hit, "ok": ok, "seconds": seconds})
    answered = [r for r in results if "ok" in r]
    print(f"  正解 {correct}/{len(results)}")

    print(f"--- 応答時間 ({n_latency} 回、同じ場面)")
    state = state_of(build_cases()[2])
    times = []
    for _ in range(n_latency):
        response, seconds, error = ask(client, state, questions, **kwargs)
        if error:
            print(f"  失敗 {error}")
            continue
        times.append(seconds)
        tokens += response.usage.input_tokens or 0
    latency = summarize_latency(times)
    if latency:
        print("  " + " ".join(f"{k}={v * 1000:.0f}ms" for k, v in latency.items()) + f" (成功 {len(times)}/{n_latency})")
    print(f"  入力トークン合計 {tokens} (約 ${tokens * INPUT_USD_PER_TOKEN:.5f})")
    return {"cases": results, "correct": correct, "answered": len(answered), "latency": latency, "input_tokens": tokens}


def summarize_latency(times: Sequence[float]) -> Dict[str, float]:
    if not times:
        return {}
    return {"min": min(times), "p50": statistics.median(times), "p95": percentile(times, 0.95), "max": max(times)}


def load_game_frame(path: Path) -> np.ndarray:
    """ゲーム画面の画像を、自機エリアだけのRGB画像 (450x384) にする。保存済みのグレー全体画像も受け付ける"""
    image = PILImage.open(path)
    x0, y0, x1, y1 = PLAYFIELD_RECT
    if image.mode == "L" and image.size == (640, 480):
        return np.asarray(image.convert("RGB"))[y0:y1, x0:x1]
    return preprocess(np.asarray(image.convert("RGB")), FrameSpec(use_rgb=True))[0]


def encode_png(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    PILImage.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def draw_box(frame: np.ndarray, box: Any, label: str, color: Tuple[int, int, int]) -> None:
    """Box の座標 (画像の高さ・幅に対する割合) を画像に描く"""
    h, w = frame.shape[:2]
    p1, p2 = (int(box.left * w), int(box.top * h)), (int(box.right * w), int(box.bottom * h))
    cv2.rectangle(frame, p1, p2, color, 1)
    cv2.putText(frame, label, (p1[0], max(p1[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


IMAGE_QUESTIONS = {
    "player": Box(instructions="The player character: a small shrine maiden in red and white near the bottom of the screen."),
    "enemy": Box(instructions="The largest enemy character or boss."),
    "move": Choice(
        instructions="Which way should the player character move horizontally to avoid the enemy bullets?",
        criteria={"left": "Move left.", "stay": "Stay.", "right": "Move right."}),
}


def run_images(client: Any, paths: Sequence[Path], n_latency: int, model: Optional[str], out_dir: Path) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"extra_headers": IMAGE_OPT_IN_HEADERS}
    if model:
        kwargs["model"] = model
    out_dir.mkdir(parents=True, exist_ok=True)
    results, tokens = [], 0
    for path in paths:
        frame = load_game_frame(path)
        response, seconds, error = ask(client, Image(encode_png(frame)), IMAGE_QUESTIONS, **kwargs)
        if error:
            print(f"  {path.name}: 失敗 {error}")
            results.append({"image": path.name, "error": error})
            continue
        annotated = frame.copy()
        item: Dict[str, Any] = {"image": path.name, "seconds": seconds}
        for name, color in (("player", (0, 255, 0)), ("enemy", (255, 0, 0))):
            answer = response.boxes[name]
            draw_box(annotated, answer.box, f"{name} {answer.confidence or 0:.2f}", color)
            item[name] = {"top": answer.box.top, "bottom": answer.box.bottom, "left": answer.box.left,
                          "right": answer.box.right, "confidence": answer.confidence}
        move = response.choices["move"]
        item["move"] = {"choice": move.choice, "probabilities": move.probabilities}
        tokens += response.usage.input_tokens or 0
        PILImage.fromarray(annotated).save(out_dir / f"annotated_{path.stem}.png")
        print(f"  {path.name}: move={move.choice} {seconds * 1000:.0f}ms 入力トークン {response.usage.input_tokens}")
        results.append(item)

    latency: Dict[str, float] = {}
    if paths and any("seconds" in r for r in results):
        frame = Image(encode_png(load_game_frame(paths[0])))
        times = [t for _, t, e in (ask(client, frame, IMAGE_QUESTIONS, **kwargs) for _ in range(n_latency)) if e is None]
        latency = summarize_latency(times)
        if latency:
            print("--- 画像の応答時間: " + " ".join(f"{k}={v * 1000:.0f}ms" for k, v in latency.items())
                  + f" (成功 {len(times)}/{n_latency})")
    print(f"  入力トークン合計 {tokens} (約 ${tokens * INPUT_USD_PER_TOKEN:.5f})")
    return {"images": results, "latency": latency, "input_tokens": tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description="Jev の実測プローブ")
    parser.add_argument("--stage", nargs="+", choices=["models", "text", "lanes", "oracle", "image"], default=["models", "text"])
    parser.add_argument("--images", type=Path, nargs="*", default=[], help="image 段階で送るゲーム画面")
    parser.add_argument("--model", default=None, help="text 段階のモデル (省略時はクライアント既定)")
    parser.add_argument("--image_model", default=None, help="image 段階のモデル (省略時はクライアント既定)")
    parser.add_argument("--per_class", type=int, default=20, help="oracle 段階の、正解クラスごとの場面数")
    parser.add_argument("--n_latency", type=int, default=20, help="応答時間の測定回数")
    parser.add_argument("--timeout", type=float, default=10.0, help="1回の呼び出しのタイムアウト (秒)")
    parser.add_argument("--out_dir", type=Path, default=Path("jev_probe_out"))
    args = parser.parse_args()

    if "image" in args.stage and not args.images:
        raise SystemExit("--stage image には --images でゲーム画面を指定してください (指定した画像だけを送ります)。")

    # リトライを無効にして、遅延や失敗 (429/5xx など) をそのまま測る
    try:
        client = TypeSafeClient(retry=RetryPolicy(max_retries=0, timeout=args.timeout))
    except TypeSafeError as error:
        raise SystemExit(f"クライアントを作れません (環境変数 TYPESAFE_API_KEY を確認してください): {type(error).__name__}")

    report: Dict[str, Any] = {}
    if "models" in args.stage:
        print("=== models")
        report["models"] = run_models(client)
    if "text" in args.stage:
        print("=== text")
        report["text"] = run_text(client, args.n_latency, args.model)
    if "lanes" in args.stage:
        print("=== lanes")
        report["lanes"] = run_lanes(client, args.n_latency, args.model)
    if "oracle" in args.stage:
        print("=== oracle")
        report["oracle"] = run_oracle(client, args.per_class, args.model)
    if "image" in args.stage:
        print("=== image")
        report["image"] = run_images(client, args.images, min(args.n_latency, 5), args.image_model, args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"結果を保存しました: {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
