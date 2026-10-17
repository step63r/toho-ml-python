"""
同じ画面を写した複数のスクリーンショットから、背景に依存しない画面判定マスクを作る

全ての画像で白い画素と全ての画像で暗い画素だけを残すため、背景の違う画像を混ぜるほど
背景に強くなる。画像は tools.capture の保存画像 (クライアント領域のRGB) か、
自機エリアだけを保存した旧形式 (384x450) のグレー画像を指定できる。

    python -m tools.build_template game_over --box 42 196 120 222 --align 4 画像1.png 画像2.png
"""
import argparse
import numpy as np

from pathlib import Path
from PIL import Image
from typing import Sequence, Tuple

from environment.capture import FrameSpec, GAME_HEIGHT, GAME_WIDTH, preprocess
from environment.screens import (
    MASK_DARK, MASK_DARK_THRESHOLD, MASK_IGNORE, MASK_WHITE, MASK_WHITE_THRESHOLD, PLAYFIELD_RECT, TEMPLATE_DIR,
    Rect, build_mask)
from utils import cv2helper

# 白画素の最低数。これより少ないと画面を区別する根拠として弱すぎる
MIN_MASK_PIXELS = 30


def load_game_gray(path: Path) -> np.ndarray:
    """画像を、画面判定と同じ 640x480 のグレー画像にする"""
    image = np.asarray(Image.open(path).convert("RGB"))
    x0, y0, x1, y1 = PLAYFIELD_RECT
    if image.shape[:2] == (y1 - y0, x1 - x0):
        canvas = np.zeros((GAME_HEIGHT, GAME_WIDTH), dtype=np.uint8)
        canvas[y0:y1, x0:x1] = image[..., 0]
        return canvas
    return preprocess(image, FrameSpec())[1]


def crop(gray: np.ndarray, box: Rect, dx: int = 0, dy: int = 0) -> np.ndarray:
    x0, y0, x1, y1 = box
    return gray[y0 + dy:y1 + dy, x0 + dx:x1 + dx]


def best_shift(reference: np.ndarray, gray: np.ndarray, box: Rect, search: int) -> Tuple[int, int]:
    """reference の白・暗画素と最も重なる位置のずれ (dx, dy) を返す"""
    ref_white, ref_dark = reference >= MASK_WHITE_THRESHOLD, reference <= MASK_DARK_THRESHOLD
    best = (-1, 0, 0)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            candidate = crop(gray, box, dx, dy)
            if candidate.shape != reference.shape:
                continue
            agree = int(np.count_nonzero(ref_white & (candidate >= MASK_WHITE_THRESHOLD))
                        + np.count_nonzero(ref_dark & (candidate <= MASK_DARK_THRESHOLD)))
            if agree > best[0]:
                best = (agree, dx, dy)
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="画面判定マスクの作成")
    parser.add_argument("name", help="マスク名。mask_<name>.png として保存する")
    parser.add_argument("images", type=Path, nargs="+", help="同じ画面を写した画像。最初の1枚が位置の基準")
    parser.add_argument("--box", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"),
                        help="切り出す範囲 (640x480 での座標)")
    parser.add_argument("--align", type=int, default=0, help="2枚目以降の位置合わせの探索幅 (px)。0なら合わせない")
    parser.add_argument("--thin_only", type=int, default=0,
                        help="0 より大きいと、その大きさより太い白い塊 (背景の輪など) を除いて、細い文字だけを残す")
    parser.add_argument("--out_dir", type=Path, default=TEMPLATE_DIR, help="保存先")
    args = parser.parse_args()

    box: Rect = tuple(args.box)  # type: ignore[assignment]
    grays = [load_game_gray(path) for path in args.images]
    reference = crop(grays[0], box)
    crops = [reference]
    for path, gray in zip(args.images[1:], grays[1:]):
        dx, dy = best_shift(reference, gray, box, args.align) if args.align else (0, 0)
        print(f"{path.name}: 位置ずれ ({dx}, {dy})")
        crops.append(crop(gray, box, dx, dy))

    mask = build_mask(crops, args.thin_only)
    white, dark = int(np.count_nonzero(mask == MASK_WHITE)), int(np.count_nonzero(mask == MASK_DARK))
    print(f"白 {white} 画素 / 暗 {dark} 画素 / 無視 {int(np.count_nonzero(mask == MASK_IGNORE))} 画素 (計 {mask.size})")
    if white < MIN_MASK_PIXELS:
        raise SystemExit(f"白画素 (文字) が {MIN_MASK_PIXELS} 未満です。範囲を見直すか、"
                         "文字が薄いフレーム (フェード中) を除いてください。")

    out_path = args.out_dir / f"mask_{args.name}.png"
    if not cv2helper.imwrite(str(out_path), mask):
        raise SystemExit(f"保存に失敗しました: {out_path}")
    print(f"保存しました: {out_path}")


if __name__ == "__main__":
    main()
