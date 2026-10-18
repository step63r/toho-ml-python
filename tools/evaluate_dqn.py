"""
学習した方策 (またはランダム方策) を実機で動かして、生存時間などを測る (ゲームにキー入力を送ります)

学習中の ep_len_mean は、探索 (ランダムな行動) が混ざった値。学習した方策そのものの強さは、ここで測る。
ランダム方策 (--random) と同じ条件で測って比べる。ゲームを前面に置き、他の操作をしないこと。Ctrl+C で中断すると、そこまでの結果を保存する。

    python -m tools.evaluate_dqn --model train_out\\default\\interrupted.zip --episodes 20
    python -m tools.evaluate_dqn --random --episodes 20

測るもの: エピソードごとのステップ数・秒数 (復旧の時間を除いた、ステップの時間の合計)・報酬・イベント
(チャプター・スペルカード・ステージクリア)、選んだ行動の偏り (同じ行動ばかり選んでいないか)。
"""
import argparse
import json
import statistics
import time

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv

from tools.train_dqn import build_vec_env


def evaluate(venv: VecEnv, episodes: int, policy: Callable[[np.ndarray], int],
             clock: Callable[[], float] = time.perf_counter) -> Dict[str, Any]:
    """
    policy (観測 → 行動) で、episodes 回のエピソードを動かす

    VecEnv は、エピソードが終わると自動でリセットする (遷移制御による復旧で、時間がかかる)。
    終わったステップの時間には、その復旧が含まれるので、足さない。代わりに、そのエピソードの他のステップの
    平均を 1 ステップ分足す。KeyboardInterrupt では、そこまでの結果を返す。
    """
    results: List[Dict[str, Any]] = []
    actions: Counter = Counter()
    interrupted = False
    obs = venv.reset()
    step_seconds: List[float] = []
    steps, events = 0, []
    try:
        while len(results) < episodes:
            action = int(policy(obs))
            actions[action] += 1
            started = clock()
            obs, _, dones, infos = venv.step(np.array([action]))
            elapsed = clock() - started
            steps += 1
            events.extend(infos[0].get("events", []))
            if not dones[0]:
                step_seconds.append(elapsed)
                continue
            seconds = sum(step_seconds) + (statistics.fmean(step_seconds) if step_seconds else 0.0)
            episode = infos[0].get("episode", {})
            results.append({"steps": int(episode.get("l", steps)), "seconds": seconds,
                            "reward": float(episode.get("r", 0.0)), "events": events})
            print(f"  episode {len(results) - 1}: {results[-1]['steps']:4d} steps  {seconds:5.1f}s  "
                  f"reward={results[-1]['reward']:g} events={events}")
            step_seconds, steps, events = [], 0, []
    except KeyboardInterrupt:
        interrupted = True
        print("中断しました。ここまでの結果を保存します。")
    return summarize(results, actions, interrupted)


def summarize(results: List[Dict[str, Any]], actions: Counter, interrupted: bool = False) -> Dict[str, Any]:
    steps = [r["steps"] for r in results]
    seconds = [r["seconds"] for r in results]
    counts: Dict[str, int] = {}
    for r in results:
        for event in r["events"]:
            counts[event] = counts.get(event, 0) + 1
    total = sum(actions.values())
    return {
        "episodes": results, "interrupted": interrupted,
        "steps": {"median": statistics.median(steps), "mean": statistics.fmean(steps), "min": min(steps),
                  "max": max(steps)} if steps else {},
        "seconds": {"median": statistics.median(seconds), "mean": statistics.fmean(seconds), "min": min(seconds),
                    "max": max(seconds)} if seconds else {},
        "event_counts": counts,
        # 上位の行動の割合。1 つの行動に偏っていれば、学習が崩れている (全 18 通りが均等なら各 5.6%)
        "top_actions": [(a, n / total) for a, n in actions.most_common(5)] if total else [],
    }


def main() -> None:
    from environment.shanghai_alice_15th import Kanjuden
    from tools.common import add_window_args, open_window

    parser = argparse.ArgumentParser(description="学習した方策 (またはランダム方策) を測る (ゲームにキー入力を送ります)")
    add_window_args(parser)
    parser.set_defaults(shrink_ratio=0.25)
    parser.add_argument("--model", type=Path, default=None, help="学習したモデル (.zip)")
    parser.add_argument("--random", action="store_true", help="ランダム方策 (比較の基準)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="学習した方策にランダムな行動を混ぜる割合")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--n_stack", type=int, default=4)
    parser.add_argument("--step_period_sec", type=float, default=None,
                        help="学習したときと同じ値を指定する (固定周期のモード。ランダム方策と比べるときは、両方に同じ値を指定する)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=Path, default=Path("train_out") / "eval")
    args = parser.parse_args()

    if args.random == (args.model is not None):
        parser.error("--model か --random の、どちらか一方を指定してください。")
    if args.model is not None and not args.model.is_file():
        parser.error(f"モデルが見つかりません: {args.model}")

    open_window(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def env_factory():
        return Kanjuden(app_class_name=args.app_class_name, app_window_name=args.app_window_name,
                        shrink_ratio=args.shrink_ratio, clip_top=args.clip_top, clip_bottom=args.clip_bottom,
                        clip_left=args.clip_left, clip_right=args.clip_right, step_period_sec=args.step_period_sec)

    venv = build_vec_env(env_factory, args.out_dir, args.n_stack)
    rng = np.random.default_rng()
    model: Optional[DQN] = None
    if args.model is not None:
        model = DQN.load(args.model, env=venv, device=args.device)
        venv = model.get_env()                       # 観測を、モデルの形 (チャンネルが先) にする包みの付いた環境

    def policy(obs: np.ndarray) -> int:
        if model is None or rng.random() < args.epsilon:
            return int(rng.integers(venv.action_space.n))
        return int(model.predict(obs, deterministic=True)[0][0])

    label = "ランダム方策" if model is None else f"{args.model} (ステップ数 {model.num_timesteps}, ε={args.epsilon})"
    print(f"=== 評価: {label} (ゲームにキー入力を送ります。Ctrl+C で中断)")
    try:
        report = evaluate(venv, args.episodes, policy)
    finally:
        venv.close()                                 # 押しっぱなしのキーを離す

    print("--- 概要")
    for key in ("steps", "seconds"):
        if report[key]:
            print(f"  {key}: " + " ".join(f"{k}={v:.1f}" for k, v in report[key].items()))
    print(f"  イベント: {report['event_counts']}")
    print("  選んだ行動の上位: " + ", ".join(f"{a}={share:.0%}" for a, share in report["top_actions"]))
    name = "random" if model is None else args.model.stem
    out = args.out_dir / f"eval_{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"結果を保存しました: {out}")


if __name__ == "__main__":
    main()
