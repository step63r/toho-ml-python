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
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv

from environment.demo import find_episodes, load_session, playable_mask
from tools.train_dqn import build_vec_env


def evaluate(venv: VecEnv, episodes: int, policy: Callable[[np.ndarray], int],
             clock: Callable[[], float] = time.perf_counter, max_steps: Optional[int] = None) -> Dict[str, Any]:
    """
    policy (観測 → 行動) で、episodes 回のエピソードを動かす

    VecEnv は、エピソードが終わると自動でリセットする (遷移制御による復旧で、時間がかかる)。
    終わったステップの時間には、その復旧が含まれるので、足さない。代わりに、そのエピソードの他のステップの
    平均を 1 ステップ分足す。KeyboardInterrupt では、そこまでの結果を返す。

    max_steps を指定すると、死なないまま max_steps に達したエピソードは、打ち切り (censored) として記録して、
    ゲームはそのまま続け、次のエピソードの数え直しにする (止まり続ける方策などで、エピソードが終わらないのを避ける)。
    打ち切りのステップ数は「max_steps 以上」の意味なので、平均や中央値は、実際より低く出る。
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
            censored = not dones[0] and max_steps is not None and steps >= max_steps
            if not dones[0]:
                step_seconds.append(elapsed)
                if not censored:
                    continue
            seconds = sum(step_seconds) + (statistics.fmean(step_seconds) if step_seconds and dones[0] else 0.0)
            episode = infos[0].get("episode", {})
            results.append({"steps": steps if censored else int(episode.get("l", steps)), "seconds": seconds,
                            "reward": 0.0 if censored else float(episode.get("r", 0.0)), "events": events,
                            "censored": censored})
            print(f"  episode {len(results) - 1}: {results[-1]['steps']:4d} steps{'+' if censored else ' '} {seconds:5.1f}s  "
                  f"reward={results[-1]['reward']:g} events={events}" + ("  (打ち切り)" if censored else ""))
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
        "episodes": results, "interrupted": interrupted, "censored": sum(1 for r in results if r.get("censored")),
        "steps": {"median": statistics.median(steps), "mean": statistics.fmean(steps), "min": min(steps),
                  "max": max(steps)} if steps else {},
        "seconds": {"median": statistics.median(seconds), "mean": statistics.fmean(seconds), "min": min(seconds),
                    "max": max(seconds)} if seconds else {},
        "event_counts": counts,
        # 上位の行動の割合。1 つの行動に偏っていれば、学習が崩れている (全 18 通りが均等なら各 5.6%)
        "top_actions": [(a, n / total) for a, n in actions.most_common(5)] if total else [],
    }


def action_prior(session_dirs: Sequence[Path], actions: int = 18) -> np.ndarray:
    """人のプレイの記録 (プレイ中のフレーム) での、18 通りの行動の割合"""
    counts = np.zeros(actions)
    for directory in session_dirs:
        session = load_session(directory)
        counts += np.bincount(session["action"][playable_mask(session)].astype(int), minlength=actions)
    if counts.sum() == 0:
        raise ValueError(f"プレイ中のフレームがありません: {list(session_dirs)}")
    return counts / counts.sum()


def sampling_policy(probabilities: np.ndarray, rng: np.random.Generator, sticky: float = 0.0) -> Callable[[np.ndarray], int]:
    """
    行動の割合 probabilities から選ぶ方策。sticky の確率で、前の行動を続ける (人は、キーを押しっぱなしにするので、
    毎ステップ選び直すと、細かく震える動きになる)。sticky=0 なら、毎ステップ独立に選ぶ
    """
    last = {"action": None}

    def policy(obs: np.ndarray) -> int:
        if last["action"] is None or rng.random() >= sticky:
            last["action"] = int(rng.choice(len(probabilities), p=probabilities))
        return last["action"]
    return policy


def main() -> None:
    from environment.shanghai_alice_15th import Kanjuden
    from tools.common import add_window_args, open_window

    parser = argparse.ArgumentParser(description="学習した方策 (またはランダム方策) を測る (ゲームにキー入力を送ります)")
    add_window_args(parser)
    parser.set_defaults(shrink_ratio=0.25)
    parser.add_argument("--model", type=Path, default=None, help="学習したモデル (.zip)")
    parser.add_argument("--random", action="store_true", help="ランダム方策 (比較の基準)")
    parser.add_argument("--constant_action", type=int, default=None, choices=range(18), metavar="0..17",
                        help="常に同じ行動を選ぶ方策 (比較の基準)。9 = Shift のみ (止まる)、0 = 何も押さない (Z は環境が管理)")
    parser.add_argument("--human_prior", type=Path, nargs="+", default=None, metavar="SESSION",
                        help="人のプレイの記録での行動の割合から選ぶ方策 (比較の基準)")
    parser.add_argument("--sticky", type=float, default=0.0,
                        help="--random / --human_prior で、前の行動を続ける確率 (人の押しっぱなしに近づける)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="学習した方策にランダムな行動を混ぜる割合")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=800,
                        help="死なないまま、このステップ数 (800 = 約 107 秒) に達したエピソードは、打ち切りとして次に進む")
    parser.add_argument("--n_stack", type=int, default=4)
    parser.add_argument("--step_period_sec", type=float, default=None,
                        help="学習したときと同じ値を指定する (固定周期のモード。ランダム方策と比べるときは、両方に同じ値を指定する)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=Path, default=Path("train_out") / "eval")
    args = parser.parse_args()

    chosen = [args.model is not None, args.random, args.constant_action is not None, args.human_prior is not None]
    if sum(chosen) != 1:
        parser.error("--model / --random / --constant_action / --human_prior のうち、1 つだけ指定してください。")
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

    if model is not None:
        def policy(obs: np.ndarray) -> int:
            if rng.random() < args.epsilon:
                return int(rng.integers(venv.action_space.n))
            return int(model.predict(obs, deterministic=True)[0][0])
        label, name = f"{args.model} (ステップ数 {model.num_timesteps}, ε={args.epsilon})", args.model.stem
    elif args.constant_action is not None:
        policy = lambda obs: args.constant_action
        label, name = f"常に行動 {args.constant_action}", f"constant{args.constant_action}"
    elif args.human_prior is not None:
        prior = action_prior(args.human_prior)
        policy = sampling_policy(prior, rng, args.sticky)
        label, name = f"人の行動の割合から選ぶ (sticky={args.sticky})", f"human_prior_sticky{args.sticky}"
    else:
        policy = sampling_policy(np.full(venv.action_space.n, 1 / venv.action_space.n), rng, args.sticky)
        label, name = f"ランダム方策 (sticky={args.sticky})", "random" if args.sticky == 0 else f"random_sticky{args.sticky}"
    print(f"=== 評価: {label} (ゲームにキー入力を送ります。Ctrl+C で中断)")
    try:
        report = evaluate(venv, args.episodes, policy, max_steps=args.max_steps)
    finally:
        venv.close()                                 # 押しっぱなしのキーを離す

    print("--- 概要")
    for key in ("steps", "seconds"):
        if report[key]:
            print(f"  {key}: " + " ".join(f"{k}={v:.1f}" for k, v in report[key].items()))
    print(f"  イベント: {report['event_counts']}  打ち切り (死なずに {args.max_steps} ステップ): {report['censored']}")
    print("  選んだ行動の上位: " + ", ".join(f"{a}={share:.0%}" for a, share in report["top_actions"]))
    out = args.out_dir / f"eval_{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"結果を保存しました: {out}")


if __name__ == "__main__":
    main()
