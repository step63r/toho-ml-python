"""
tools.record_demo で記録した人のプレイを、確認する

フレーム数・時間・撮影周期の乱れ・エピソード (死亡・ステージクリア・中断) の数・チャプターなどの出来事・押したキーの偏りを表示する。
学習に使える記録かを、判断するためのもの。--sheet で、プレイ中の画面を何枚か並べた画像も書き出す。

    python -m tools.inspect_demo demo_out\\stage1_a
    python -m tools.inspect_demo demo_out\\stage1_a demo_out\\stage1_b --sheet demo_out\\sheet.png
"""
import argparse

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from environment.demo import (FLAG_CHAPTER_FINISH, FLAG_GAME_OVER, FLAG_LOST, FLAG_SPELL_CARD_BONUS, FLAG_STAGE_CLEAR, SCREEN_NAMES,
                              find_episodes, iter_parts, load_session, playable_mask)
from utils.native import COMMAND_DICT


def action_label(action: int) -> str:
    names = {1: "左", 2: "上", 3: "右", 4: "下", 5: "左上", 6: "右上", 7: "右下", 8: "左下"}
    return f"{'Shift+' if action >= 9 else ''}{names.get(action % 9, '無し')}"


def summarize(session: Dict[str, np.ndarray], period: float) -> Dict[str, Any]:
    """記録の統計を求める (表示と分けて、テストできるようにしている)"""
    t = session["t"]
    intervals = np.diff(t) if len(t) > 1 else np.array([])
    episodes = find_episodes(session)
    actions = session["action"]
    playable = playable_mask(session, episodes)
    counts = Counter(actions[playable].tolist())
    flags = session["flags"]
    screens = Counter(SCREEN_NAMES[i] for i in session["screen"])
    return {
        "frames": int(len(t)),
        "seconds": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "period_mean": float(intervals.mean()) if len(intervals) else 0.0,
        "period_p95": float(np.percentile(intervals, 95)) if len(intervals) else 0.0,
        "late_frames": int((intervals > period * 1.5).sum()),
        "screens": dict(screens),
        "episodes": episodes,
        "playable_frames": int(playable.sum()),
        "actions": counts,
        "idle_share": counts.get(0, 0) / max(sum(counts.values()), 1),
        "z_fraction": float(session["z_fraction"][playable].astype(np.float32).mean()) if playable.any() else 0.0,
        "x_pressed_frames": int(session["x_pressed"][session["action"] >= 0].sum()),
        "esc_pressed_frames": int(session["esc_pressed"].sum()),
        "chapter_finish": int(((flags & FLAG_CHAPTER_FINISH) > 0).sum()),
        "spell_card_bonus": int(((flags & FLAG_SPELL_CARD_BONUS) > 0).sum()),
        "stage_clear": int(((flags & FLAG_STAGE_CLEAR) > 0).sum()),
        "lost_frames": int(((flags & FLAG_LOST) > 0).sum()),
        "game_over_frames": int(((flags & FLAG_GAME_OVER) > 0).sum()),
    }


def warnings(stats: Dict[str, Any], period: float) -> List[str]:
    found = []
    if stats["frames"] and stats["late_frames"] / stats["frames"] > 0.05:
        found.append(f"撮影が周期より大きく遅れたフレームが {stats['late_frames']} 件 (5% 超)。他のアプリの負荷や、画面取得の遅れが考えられる。")
    if abs(stats["period_mean"] - period) > period * 0.15:
        found.append(f"平均の周期が {stats['period_mean']:.3f} 秒で、設定の {period} 秒とずれている。")
    if stats["x_pressed_frames"]:
        found.append(f"X (ボム) が {stats['x_pressed_frames']} フレームで押されている。環境の行動にボムは無いので、そのフレームは使わない方がよい。")
    if stats["playable_frames"] and stats["z_fraction"] < 0.5:
        found.append(f"Z (ショット) を押していた割合が {stats['z_fraction']:.0%} と低い。ショットを押しっぱなしにして遊んでいるか確認する。")
    if stats["playable_frames"] and stats["idle_share"] > 0.6:
        found.append(f"何も押していない (または方向を押していない) フレームが {stats['idle_share']:.0%}。キーが読めていない可能性がある。")
    if stats["lost_frames"] > 200:
        found.append(f"静止で「迷子」と判定されたフレームが {stats['lost_frames']} 件 (メニュー・ポーズなど)。そのフレームは、エピソードに含めない扱いになる。")
    return found


def sheet(session_dirs: Sequence[Path], output: Path, count: int = 12) -> None:
    """プレイ中の画面を、ランダムに並べた画像を書く (0.25 倍・等倍の左右比較)。画像の読み込みは 1 part ずつ"""
    import cv2
    rng = np.random.default_rng(0)
    tiles = []
    for directory in session_dirs:
        for part in iter_parts(directory):
            playing = np.where(part["screen"] == SCREEN_NAMES.index("playing"))[0]
            if len(playing) == 0:
                continue
            for i in rng.choice(playing, size=min(2, len(playing)), replace=False):
                tile = cv2.cvtColor(part["obs"][i], cv2.COLOR_GRAY2BGR)
                cv2.putText(tile, action_label(int(part["action"][i])), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                tiles.append(cv2.resize(tile, (192, 225)))
    tiles = tiles[:count]
    if not tiles:
        return
    columns = 4
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.vstack(rows))[1].tofile(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="記録した人のプレイを確認する")
    parser.add_argument("sessions", type=Path, nargs="+", help="tools.record_demo の出力ディレクトリ")
    parser.add_argument("--period", type=float, default=0.133, help="記録したときの周期 (秒)")
    parser.add_argument("--sheet", type=Path, default=None, help="プレイ中の画面を並べた画像の出力先")
    args = parser.parse_args()

    total = Counter()
    for directory in args.sessions:
        stats = summarize(load_session(directory), args.period)
        reasons = Counter(e["reason"] for e in stats["episodes"])
        print(f"=== {directory}")
        print(f"  フレーム {stats['frames']} ({stats['seconds'] / 60:.1f} 分)  周期 平均 {stats['period_mean'] * 1000:.0f} ms / 95% {stats['period_p95'] * 1000:.0f} ms / 遅れ {stats['late_frames']} 件")
        print(f"  エピソード {len(stats['episodes'])} (死亡 {reasons['death']} / ステージクリア {reasons['clear']} / 中断 {reasons['left']} / 記録の終わり {reasons['open']})"
              f"  プレイ中のフレーム {stats['playable_frames']} ({stats['playable_frames'] * args.period / 60:.1f} 分)")
        print(f"  チャプター {stats['chapter_finish']}  スペルカードボーナス {stats['spell_card_bonus']}  ステージクリア {stats['stage_clear']}")
        top = ", ".join(f"{action_label(a)} {n / max(sum(stats['actions'].values()), 1):.0%}" for a, n in stats["actions"].most_common(6))
        print(f"  押したキー (プレイ中、上位): {top}")
        print(f"  Z を押していた割合 {stats['z_fraction']:.0%}  X (ボム) {stats['x_pressed_frames']} フレーム  Esc {stats['esc_pressed_frames']} フレーム")
        print(f"  画面の内訳: " + ", ".join(f"{k} {v}" for k, v in sorted(stats["screens"].items(), key=lambda kv: -kv[1])[:6]))
        for message in warnings(stats, args.period):
            print(f"  注意: {message}")
        total.update(frames=stats["frames"], playable=stats["playable_frames"], deaths=reasons["death"])
    if len(args.sessions) > 1:
        print(f"--- 合計: フレーム {total['frames']}  プレイ中 {total['playable']} ({total['playable'] * args.period / 60:.1f} 分)  死亡 {total['deaths']}")
    if args.sheet:
        sheet(args.sessions, args.sheet)
        print(f"画像を書きました: {args.sheet}")


if __name__ == "__main__":
    main()
