import argparse
import logging

from environment.capture import DEFAULT_APP_CLASS_NAME, DEFAULT_APP_WINDOW_NAME, FrameSpec, GameWindow


def add_window_args(parser: argparse.ArgumentParser) -> None:
    """main.py と同じウィンドウ・切り取り引数を追加する"""
    spec = FrameSpec()
    parser.add_argument("--app_class_name", "-c", type=str, default=DEFAULT_APP_CLASS_NAME,
                        help="キャプチャ対象のウィンドウクラス名")
    parser.add_argument("--app_window_name", "-w", type=str, default=DEFAULT_APP_WINDOW_NAME,
                        help="キャプチャ対象のウィンドウ名")
    parser.add_argument("--shrink_ratio", type=float, default=spec.shrink_ratio, help="観測画像の縮小率")
    parser.add_argument("--clip_top", type=int, default=spec.clip_top, help="画像を切り取る位置 (top, 640x480 での座標)")
    parser.add_argument("--clip_bottom", type=int, default=spec.clip_bottom, help="画像を切り取る位置 (bottom)")
    parser.add_argument("--clip_left", type=int, default=spec.clip_left, help="画像を切り取る位置 (left)")
    parser.add_argument("--clip_right", type=int, default=spec.clip_right, help="画像を切り取る位置 (right)")


def open_window(args: argparse.Namespace) -> GameWindow:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    spec = FrameSpec(shrink_ratio=args.shrink_ratio, clip_top=args.clip_top, clip_bottom=args.clip_bottom,
                     clip_left=args.clip_left, clip_right=args.clip_right)
    return GameWindow(args.app_class_name, args.app_window_name, spec)
