"""
人のプレイを記録する (計画 5: 人のプレイの記録による事前学習)

ゲームの画面を、学習と同じ周期 (既定 0.133 秒) と同じ判定で撮りながら、あなたが押しているキー (方向・Shift) を記録する。
このツールは、ゲームにキーを一切送らない (読むだけ)。ゲームの開始 (メニュー操作) も、あなたが手で行う。

手順:
  1. ゲームを起動しておく。ゲームを前面に置き、他のウィンドウで隠さない。
  2. このツールを実行する (ターミナルは、ゲームを隠さない場所に置く)。「記録中」と出る。
  3. ゲームで、ステージ 1 の最初から始める (タイトル → Game Start → モード・難易度・キャラの選択 → 中断データの確認は
     「いいえ」(最初から) を選ぶ)。メニューの間の画面も記録されるが、学習には使わない。
  4. 普通に遊ぶ。死んだら、いつもどおりリトライしてよい。ポーズ (Esc) も使ってよい (その間は使わない区間として扱う)。
  5. 終わるとき: Ctrl+C か、出力ディレクトリに STOP ファイルを作る (ゲームを前面に置いたまま止められる)。
         New-Item demo_out\\<セッション名>\\STOP

    python -m tools.record_demo
    python -m tools.record_demo --out_dir demo_out\\stage1_a --max_minutes 30

記録したものの確認: python -m tools.inspect_demo demo_out\\<セッション名>

注意:
  - 使うキーは、ゲームの既定 (方向キー・Z ショット・Shift 低速)。ボム (X) は押されても記録するが、環境の行動には無い。
  - 環境の行動は、方向 (8 方向 + 無し) × Shift の 18 通り。Z は、環境が管理するので、行動には含めない。
  - ゲームのウィンドウの位置・大きさを、記録中に変えないこと。
"""
import argparse
import logging
import time

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from environment.demo import (EVENT_FLAGS, FLAG_LOST, DemoWriter, KeySampler, SCREEN_NAMES, summarize_interval)

logger = logging.getLogger(__name__)

STOP_FILE_NAME = "STOP"
DEFAULT_PERIOD_SEC = 0.133
# 画面が判定できない (unknown) 状態がこの秒数続いたら、ゲームが他のウィンドウで隠れていないかを知らせる。
# ローディングなどの短い unknown ではなく、撮れているのがゲームでないことを疑う長さ
UNKNOWN_WARNING_SEC = 15.0
UNKNOWN_WARNING_INTERVAL_SEC = 30.0


def record(env: Any, sampler: KeySampler, writer: DemoWriter, period: float = DEFAULT_PERIOD_SEC,
           should_stop: Callable[[], bool] = lambda: False, max_seconds: Optional[float] = None,
           clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
           status: Callable[[str], None] = lambda message: None, status_every: float = 5.0) -> Dict[str, Any]:
    """
    周期ごとに、画面を撮って判定し (env.observe())、前の周期のキーの状態から行動を決めて、writer に書く

    遅れたときは、遅れを取り戻そうとして詰めて撮らず、次の周期にそろえる (周期が乱れた回数は、結果に残す)。
    """
    started = clock()
    next_tick = started
    previous: Optional[float] = None
    periods = []
    counts = {"deaths": 0, "chapter_finish": 0, "spell_card_bonus": 0, "stage_clear": 0}
    last_status = started
    screen = "?"
    died_before = False
    unknown_since: Optional[float] = None
    last_unknown_warning = -UNKNOWN_WARNING_INTERVAL_SEC
    try:
        while not should_stop() and (max_seconds is None or clock() - started < max_seconds):
            wait = next_tick - clock()
            if wait > 0:
                sleep(wait)
            now = clock()
            observation, reward, terminated, _, info = env.observe()
            if previous is not None:
                writer.set_action(**summarize_interval(sampler.consume(previous, now), previous, now))
                periods.append(now - previous)
            flags = sum(EVENT_FLAGS.get(event, 0) for event in info.get("events", []))
            flags |= FLAG_LOST if info.get("lost") else 0
            screen = info.get("screen", "unknown")
            writer.add(observation, now, screen, float(reward), bool(terminated), flags, float(info.get("game_over_score", 0.0)))
            for event in ("chapter_finish", "spell_card_bonus", "stage_clear"):
                counts[event] += event in info.get("events", [])
            unknown_since = (now if unknown_since is None else unknown_since) if screen == "unknown" else None
            if (unknown_since is not None and now - unknown_since >= UNKNOWN_WARNING_SEC
                    and now - last_unknown_warning >= UNKNOWN_WARNING_INTERVAL_SEC):
                status(f"注意: 画面が {now - unknown_since:.0f} 秒間、判定できません。ゲームが他のウィンドウで隠れていないか、"
                       "ゲームのウィンドウを動かしていないか確認してください (隠れていると、記録は使えません)。")
                last_unknown_warning = now
            died = screen == "game_over"
            counts["deaths"] += died and not died_before
            died_before = died
            previous = now
            # 遅れたとき (撮影が周期の半分以上遅れた) は、遅れを取り戻そうとして間隔を詰めず、今から一周期後にそろえ直す
            next_tick = now + period if now - next_tick > period * 0.5 else next_tick + period
            if now - last_status >= status_every:
                status(f"経過 {int(now - started) // 60:02d}:{int(now - started) % 60:02d}  フレーム {writer.frames}  画面 {screen:16s} "
                       f"死亡 {counts['deaths']}  チャプター {counts['chapter_finish']}  スペル {counts['spell_card_bonus']}  "
                       f"ステージクリア {counts['stage_clear']}")
                last_status = now
    except KeyboardInterrupt:
        status("中断しました。ここまでを保存します。")
    if previous is not None:
        now = clock()
        writer.set_action(**summarize_interval(sampler.consume(previous, now), previous, now))
    late = sum(1 for p in periods if p > period * 1.5)
    return {"frames": writer.frames, "seconds": clock() - started, "late_frames": late,
            "mean_period": sum(periods) / len(periods) if periods else 0.0, **counts}


def main() -> None:
    from environment.shanghai_alice_15th import Kanjuden
    from tools.common import add_window_args, open_window

    parser = argparse.ArgumentParser(description="人のプレイを記録する")
    add_window_args(parser)
    parser.set_defaults(shrink_ratio=1.0)           # 切り出した等倍の画像を保存する (あとで縮小率を変えても、撮り直さずに済む)
    parser.add_argument("--out_dir", type=Path, default=None, help="既定: demo_out\\<日時>")
    parser.add_argument("--period", type=float, default=DEFAULT_PERIOD_SEC, help="撮影の周期 (秒)。学習の環境と同じにする")
    parser.add_argument("--max_minutes", type=float, default=None, help="この時間で自動的に終わる")
    args = parser.parse_args()

    out_dir = args.out_dir or Path("demo_out") / time.strftime("%Y%m%d_%H%M%S")
    if (out_dir / "meta.json").exists() or list(out_dir.glob("part_*.npz")):
        parser.error(f"{out_dir} には、すでに記録があります。別の --out_dir を指定してください (上書きを避けるため)。")
    out_dir.mkdir(parents=True, exist_ok=True)
    open_window(args)

    env = Kanjuden(app_class_name=args.app_class_name, app_window_name=args.app_window_name, shrink_ratio=args.shrink_ratio,
                   clip_top=args.clip_top, clip_bottom=args.clip_bottom, clip_left=args.clip_left, clip_right=args.clip_right,
                   auto_advance=False)               # ツールは、ゲームにキーを一切送らない
    sampler = KeySampler()
    writer = DemoWriter(out_dir, meta={"period": args.period, "observation_shape": list(env.observation_space.shape),
                                       "screens": list(SCREEN_NAMES), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print("=== 記録中 (このツールは、ゲームにキーを送りません)。ゲームで、ステージ 1 の最初から始めて、遊んでください。")
    print(f"    終わるときは Ctrl+C か、{out_dir / STOP_FILE_NAME} を作成します。")

    stop_file = out_dir / STOP_FILE_NAME
    sampler.start()
    try:
        result = record(env, sampler, writer, args.period, should_stop=stop_file.exists,
                        max_seconds=None if args.max_minutes is None else args.max_minutes * 60, status=print)
    finally:
        sampler.stop()
        env.close()
        meta = writer.close()
    print(f"終了: {result}")
    print(f"保存しました: {out_dir} ({meta['frames']} フレーム、{meta['parts']} 個の part)")
    print(f"確認: python -m tools.inspect_demo {out_dir}")


if __name__ == "__main__":
    main()
