"""
今の画面から、プレイ画面になるまで自動で進める (実機での遷移制御の確認用)

タイトル・各選択画面・ゲームオーバー画面のどこから実行してもよい。猶予 (--delay) の間に
ゲームウィンドウをクリックしてフォーカスしておくこと。想定外の画面で止まった場合は
FlowError を出して終了する (その画面を tools.capture で保存して教えてほしい)。

    python -m tools.run_flow
"""
import argparse
import logging
import time

from environment.flow import GameFlow, default_flow_config
from environment.screens import ScreenDetector
from tools.common import add_window_args, open_window
from utils.native import HoldKeys


def main() -> None:
    parser = argparse.ArgumentParser(description="プレイ画面まで自動遷移")
    add_window_args(parser)
    parser.add_argument("--delay", type=float, default=5.0, help="開始までの猶予 (秒)")
    args = parser.parse_args()

    window = open_window(args)
    flow = GameFlow(
        grab_gray=lambda: window.grab()[1],
        press=lambda codes: HoldKeys(codes, 4),
        detector=ScreenDetector(),
        config=default_flow_config())

    print(f"{args.delay:.0f} 秒後に開始します。ゲームウィンドウにフォーカスしてください。")
    time.sleep(args.delay)
    window.focus()
    flow.advance_to_playing()
    logging.getLogger(__name__).info("プレイ画面になりました。")


if __name__ == "__main__":
    main()
