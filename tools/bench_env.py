"""
実機の環境 (Kanjuden) をランダム方策で動かし、ステップ速度と、学習前のベースラインを測る

測るもの:
  - 1秒あたりのステップ数と、その内訳 (画面取得・画面判定・キー入力)
  - エピソード (ゲームオーバーまたはステージクリアまで) ごとの生存時間・報酬・イベント

学習前のランダム方策の生存時間は、DQN が学習で改善したかを判断する基準になる。
ゲームを起動しておく (タイトル・プレイ中・ゲームオーバーのどこからでもよい)。実行中、ランダムなキー入力を
ゲームに送るので、ゲームウィンドウを前面のままにし、他の操作をしないこと。Ctrl+C で止めると、そこまでの結果を保存する。

    python -m tools.bench_env --episodes 10 --max_episode_seconds 120
"""
import argparse
import json
import statistics
import time

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from environment.shanghai_alice_15th import Kanjuden
from tools.common import add_window_args, open_window


class Timers:
    """環境の内部処理 (画面取得・画面判定) の所要時間を集める"""
    def __init__(self) -> None:
        self.samples: Dict[str, List[float]] = {"capture": [], "detect": []}

    def wrap(self, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self.samples[name].append(time.perf_counter() - start)
        return timed

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {name: {"mean_ms": statistics.fmean(v) * 1000, "p95_ms": sorted(v)[int(0.95 * (len(v) - 1))] * 1000,
                       "count": len(v)} for name, v in self.samples.items() if v}


def instrument(env: Kanjuden) -> Timers:
    """env の画面取得と画面判定に、時間計測を差し込む (環境の内部を直接差し替えるのは、この計測のためだけ)"""
    timers = Timers()
    env._window.grab = timers.wrap("capture", env._window.grab)
    env._detector.detect = timers.wrap("detect", env._detector.detect)
    return timers


def run_bench(env: Kanjuden, episodes: int, max_episode_seconds: float, max_seconds: float,
              policy: Optional[Callable[[Any], int]] = None) -> Dict[str, Any]:
    """方策 (省略時はランダム) で episodes 回のエピソードを回し、結果を返す"""
    policy = policy or (lambda observation: int(env.action_space.sample()))
    timers = instrument(env)
    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    interrupted = False
    try:
        for index in range(episodes):
            if time.perf_counter() - started >= max_seconds:
                break
            observation, _ = env.reset()
            episode_start, steps, total, events = time.perf_counter(), 0, 0.0, []
            ended = "time"
            while time.perf_counter() - episode_start < max_episode_seconds:
                observation, reward, terminated, _, info = env.step(policy(observation))
                steps += 1
                total += reward
                events.extend(info.get("events", []))
                if terminated:
                    ended = "stage_clear" if "stage_clear" in info.get("events", []) else "game_over"
                    break
            seconds = time.perf_counter() - episode_start
            results.append({"episode": index, "steps": steps, "seconds": seconds, "reward": total,
                            "ended": ended, "events": events})
            print(f"  episode {index}: {ended:11s} {seconds:6.1f}s {steps:5d} steps ({steps / max(seconds, 1e-9):.1f}/s) "
                  f"reward={total:g} events={events}")
    except KeyboardInterrupt:
        interrupted = True
        print("中断しました。ここまでの結果を保存します。")
        env.dump_diagnostics("interrupted")
    finally:
        env.close()
    return summarize(results, timers.summary(), time.perf_counter() - started, interrupted)


def summarize(results: Sequence[Dict[str, Any]], timing: Dict[str, Any], total_seconds: float,
              interrupted: bool = False) -> Dict[str, Any]:
    steps = sum(r["steps"] for r in results)
    seconds = sum(r["seconds"] for r in results)
    died = [r for r in results if r["ended"] == "game_over"]
    counts: Dict[str, int] = {}
    for r in results:
        for event in r["events"]:
            counts[event] = counts.get(event, 0) + 1
    return {
        "episodes": list(results),
        "interrupted": interrupted,
        "total_seconds": total_seconds,
        "steps_per_second": steps / seconds if seconds else 0.0,
        "survival_seconds": {"median": statistics.median(r["seconds"] for r in died),
                             "mean": statistics.fmean(r["seconds"] for r in died),
                             "min": min(r["seconds"] for r in died), "max": max(r["seconds"] for r in died)} if died else {},
        "ended": {kind: sum(1 for r in results if r["ended"] == kind) for kind in ("game_over", "stage_clear", "time")},
        "event_counts": counts,
        "timing": timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ランダム方策でのベースライン測定")
    add_window_args(parser)
    parser.add_argument("--episodes", type=int, default=10, help="エピソード数")
    parser.add_argument("--max_episode_seconds", type=float, default=120.0, help="1エピソードの最大時間 (秒)")
    parser.add_argument("--max_seconds", type=float, default=900.0, help="全体の最大時間 (秒)")
    parser.add_argument("--out_dir", type=Path, default=Path("bench_out"))
    parser.add_argument("--diagnostics_dir", type=Path, default=Path("diagnostics_out"),
                        help="迷子・中断のときに、直近のフレームと判定・動き量を保存するディレクトリ")
    args = parser.parse_args()

    open_window(args)  # ログ設定とウィンドウの存在確認
    env = Kanjuden(app_class_name=args.app_class_name, app_window_name=args.app_window_name,
                   shrink_ratio=args.shrink_ratio, clip_top=args.clip_top, clip_bottom=args.clip_bottom,
                   clip_left=args.clip_left, clip_right=args.clip_right,
                   diagnostics_dir=str(args.diagnostics_dir))
    print("=== ランダム方策 (ゲームにキー入力を送ります。Ctrl+C で中断)")
    report = run_bench(env, args.episodes, args.max_episode_seconds, args.max_seconds)

    print("--- 概要")
    print(f"  ステップ速度: {report['steps_per_second']:.1f} steps/s")
    for name, t in report["timing"].items():
        print(f"  {name}: 平均 {t['mean_ms']:.1f}ms / p95 {t['p95_ms']:.1f}ms ({t['count']} 回)")
    print(f"  終了の内訳: {report['ended']}  イベント: {report['event_counts']}")
    if report["survival_seconds"]:
        print("  ゲームオーバーまでの生存時間(秒): " + " ".join(f"{k}={v:.1f}" for k, v in report["survival_seconds"].items()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "baseline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"結果を保存しました: {args.out_dir / 'baseline.json'}")


if __name__ == "__main__":
    main()
