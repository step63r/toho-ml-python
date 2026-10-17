import cv2
import logging
import numpy as np

from dataclasses import dataclass
from PIL import ImageGrab
from typing import Tuple

from utils.native import FocusWindow, GetClientRect


logger = logging.getLogger(__name__)

DEFAULT_APP_CLASS_NAME = "BASE"
DEFAULT_APP_WINDOW_NAME = "東方紺珠伝　～ Legacy of Lunatic Kingdom. ver 1.00b"

# ゲーム本来の解像度。ウィンドウの表示倍率に関わらず、この大きさに正規化してから処理する
GAME_WIDTH = 640
GAME_HEIGHT = 480


@dataclass(frozen=True)
class FrameSpec:
    """
    ゲーム画面から観測画像を切り出す設定

    clip_* はゲーム本来の解像度 (640x480) での座標で、既定値は自機エリア (384x450)。
    画面判定用のグレー画像は常に等倍の全体 (640x480) で、clip_* と shrink_ratio は観測画像にだけ適用する
    (学習用に縮小しても、画面判定の精度が変わらないようにするため)。
    """
    use_rgb: bool = False
    shrink_ratio: float = 1.0
    clip_top: int = 16
    clip_bottom: int = 466
    clip_left: int = 32
    clip_right: int = 416

    @property
    def height(self) -> int:
        return self.clip_bottom - self.clip_top

    @property
    def width(self) -> int:
        return self.clip_right - self.clip_left

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        shape: Tuple[int, ...] = (int(self.height * self.shrink_ratio), int(self.width * self.shrink_ratio))
        return shape + ((3,) if self.use_rgb else ())


def preprocess(rgb_client: np.ndarray, spec: FrameSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    ゲーム画面 (クライアント領域) のRGB画像から、観測画像とテンプレート照合用のグレー画像を作る

    Parameters
    ----------
    rgb_client : numpy.ndarray
        クライアント領域の画像 (H, W, 3, RGB)。大きさは問わない
    spec : FrameSpec
        切り取り位置と観測画像の形式

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        (観測画像, グレー画像)。グレー画像は画面判定用で、切り取らない 640x480 全体
    """
    # 既存テンプレートは RGB データに BGR2GRAY をかけたグレー画像から切り出されている。
    # RGB2GRAY に直すと輝度の重みが変わって一致度が下がるため、あえて BGR2GRAY のままにする。
    gray_full = cv2.cvtColor(rgb_client, cv2.COLOR_BGR2GRAY)

    def normalize(image: np.ndarray) -> np.ndarray:
        # 縮小には INTER_AREA を使う。2倍表示なら2x2の平均になり、等倍作成のテンプレートと整合する
        return cv2.resize(image, (GAME_WIDTH, GAME_HEIGHT), interpolation=cv2.INTER_AREA)

    def clip(image: np.ndarray) -> np.ndarray:
        return image[spec.clip_top:spec.clip_bottom, spec.clip_left:spec.clip_right]

    gray = normalize(gray_full)
    observation = clip(normalize(rgb_client)) if spec.use_rgb else clip(gray)
    if spec.shrink_ratio != 1.0:
        height, width = spec.observation_shape[:2]
        observation = cv2.resize(observation, (width, height), interpolation=cv2.INTER_AREA)
    return observation, gray


class GameWindow:
    """
    ゲームウィンドウの画面取得

    ウィンドウ座標は初期化時に一度だけ取得する (これ以降ウィンドウの位置・大きさを変えないこと)。
    """
    def __init__(self, app_class_name: str = DEFAULT_APP_CLASS_NAME,
                 app_window_name: str = DEFAULT_APP_WINDOW_NAME,
                 spec: FrameSpec = FrameSpec()) -> None:
        self.app_class_name = app_class_name
        self.app_window_name = app_window_name
        self.spec = spec
        self.left, self.top, self.right, self.bottom = GetClientRect(app_class_name, app_window_name)

    def focus(self) -> bool:
        focused = FocusWindow(self.app_class_name, self.app_window_name)
        if not focused:
            logger.warning("ゲームウィンドウを前面にできませんでした。キー入力が届かない可能性があります。")
        return focused

    def grab_full(self) -> np.ndarray:
        """クライアント領域全体のRGB画像 (H, W, 3)"""
        img = ImageGrab.grab((self.left, self.top, self.right, self.bottom),
                             include_layered_windows=False, all_screens=True)
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def grab(self) -> Tuple[np.ndarray, np.ndarray]:
        """(観測画像, グレー画像) を返す"""
        return preprocess(self.grab_full(), self.spec)
