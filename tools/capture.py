"""
ゲームウィンドウ全体のスクリーンショットを captures/ に保存する

画面遷移の判定用テンプレートを作る素材集めに使う。
F8 を押すたびに1枚保存する (ゲームにフォーカスがあっても有効)。--interval を指定すると定期保存する。

    python -m tools.capture
    python -m tools.capture --interval 0.5
"""
import argparse
import time

from datetime import datetime
from pathlib import Path
from PIL import Image

from tools.common import add_window_args, open_window
from utils.native import IsKeyDown

VK_F8 = 0x77


def main() -> None:
    parser = argparse.ArgumentParser(description="ゲーム画面の保存")
    add_window_args(parser)
    parser.add_argument("--out_dir", type=Path, default=Path("captures"), help="保存先")
    parser.add_argument("--interval", type=float, default=0.0, help="定期保存の間隔 (秒)。0 ならF8押下時のみ")
    args = parser.parse_args()

    window = open_window(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"F8 で保存 / Ctrl+C で終了 (保存先: {args.out_dir.resolve()})")

    saved = 0
    was_down = False
    next_auto = time.monotonic()
    while True:
        down = IsKeyDown(VK_F8)
        due = args.interval > 0 and time.monotonic() >= next_auto
        if (down and not was_down) or due:
            name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".png"
            Image.fromarray(window.grab_full()).save(args.out_dir / name)
            saved += 1
            next_auto = time.monotonic() + args.interval
            print(f"[{saved}] {name}")
        was_down = down
        time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
