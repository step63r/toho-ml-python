import argparse

from stable_baselines3.ppo.ppo import PPO
from typing import Any

from environment.shanghai_alice_15th import Kanjuden


def main(args: Any):
    """
    メインメソッド

    Parameters
    ----------
    args : Any
        コマンドライン引数
    """
    app_class_name: str = args.app_class_name
    app_window_name: str = args.app_window_name
    use_rgb: bool = args.use_rgb
    shrink_ratio: float = args.shrink_ratio
    clip_top: int = args.clip_top
    clip_bottom: int = args.clip_bottom
    clip_left: int = args.clip_left
    clip_right: int = args.clip_right

    env = Kanjuden(
        app_class_name=app_class_name,
        app_window_name=app_window_name,
        use_rgb=use_rgb,
        shrink_ratio=shrink_ratio,
        clip_top=clip_top,
        clip_bottom=clip_bottom,
        clip_left=clip_left,
        clip_right=clip_right)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=128000)
    model.save('kanjuden')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="東方紺珠伝の深層強化学習プログラム")
    parser.add_argument(
        "--app_class_name", "-c", type=str,
        default="BASE",
        help="キャプチャ対象のウィンドウクラス名")
    parser.add_argument(
        "--app_window_name", "-w", type=str,
        default="東方紺珠伝　～ Legacy of Lunatic Kingdom. ver 1.00b",
        help="キャプチャ対象のウィンドウ名")
    parser.add_argument("--use_rgb", action="store_true", help="RGB画像を使う")
    parser.add_argument("--shrink_ratio", type=float, default=0.5, help="画像縮小率")
    parser.add_argument("--clip_top", type=int, default=26, help="画像を切り取る位置 (top)")
    parser.add_argument("--clip_bottom", type=int, default=476, help="画像を切り取る位置 (bottom)")
    parser.add_argument("--clip_left", type=int, default=32, help="画像を切り取る位置 (left)")
    parser.add_argument("--clip_right", type=int, default=416, help="画像を切り取る位置 (right)")
    args = parser.parse_args()
    main(args)
