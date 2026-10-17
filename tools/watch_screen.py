"""
画面判定の結果をリアルタイムに表示する (入力は一切送らない)

ゲームを手で遊びながら、各画面 (タイトル・選択画面・プレイ中・ゲームオーバー等) を
正しく検出できているか確認するのに使う。右端に一致率の高い画面を3つ表示する。

    python -m tools.watch_screen
"""
import argparse
import time

from environment.screens import GameOverTracker, ScreenDetector
from tools.common import add_window_args, open_window


def main() -> None:
    parser = argparse.ArgumentParser(description="画面判定のライブ表示")
    add_window_args(parser)
    parser.add_argument("--hz", type=float, default=10.0, help="判定の頻度")
    args = parser.parse_args()

    window = open_window(args)
    detector = ScreenDetector()
    tracker = GameOverTracker()
    print("Ctrl+C で終了。判定が変わった時と1秒ごとに表示します。")

    last_key = None
    last_print = 0.0
    while True:
        _, gray = window.grab()
        detection = detector.detect(gray)
        state = tracker.update(detection.game_over_score)
        key = (detection.screen, state, detection.overlays)
        now = time.monotonic()
        if key != last_key or now - last_print >= 1.0:
            top = sorted(detection.scores.items(), key=lambda item: -item[1])[:3]
            scores = " ".join(f"{name}={score:.2f}" for name, score in top)
            overlays = ",".join(sorted(o.value for o in detection.overlays)) or "-"
            print(f"screen={detection.screen.value:23s} tracker={state.value:5s} overlays={overlays:30s} {scores}")
            last_key, last_print = key, now
        time.sleep(1.0 / args.hz)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
