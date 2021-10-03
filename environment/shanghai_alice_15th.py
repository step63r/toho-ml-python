import cv2
import gym
import time
import tkinter
import numpy as np


from gym import spaces
from PIL import Image, ImageGrab, ImageTk
from typing import Any, Dict, Tuple

from numpy.lib.type_check import imag


from utils import cv2helper
from utils.native import COMMAND_DICT, SCAN_Z, SCAN_RIGHT, GetWindowRect, PressKey, ReleaseKey


class Kanjuden(gym.Env):
    """
    「東方紺珠伝」向けカスタムGym環境

        - 目的：ゲームオーバーにならない
        - 行動：ESCもしくは (Z, 8方向, Shift) の組み合わせ
        - 状態：画面イメージ
        - 報酬：
        - エピソード完了：

    Parameters
    ----------
    gym : [type]
        [description]
    """
    def __command_start(self, response: int) -> None:
        """
        キーを押下する

        Parameters
        ----------
        response : int
            サーバレスポンス
        """
        if response in COMMAND_DICT:
            for v in COMMAND_DICT[response]:
                PressKey(v)
    
    def __command_end(self, response: int) -> None:
        """
        キーを離す

        Parameters
        ----------
        response : int
            サーバレスポンス
        """
        if response in COMMAND_DICT:
            for v in COMMAND_DICT[response]:
                ReleaseKey(v)
    
    def __press_and_release_key(self, hex_key_code: int) -> None:
        """
        適当なスリープを挟んでキーを押下する

        Parameters
        ----------
        hex_key_code : int
            キーコード
        """
        PressKey(hex_key_code)
        time.sleep(4 * (1.0 / 60))
        ReleaseKey(hex_key_code)
        time.sleep(2)

    def __init__(self, grid_size=[384, 450, 1],
                 app_class_name: str = "BASE",
                 app_window_name: str = "東方紺珠伝　～ Legacy of Lunatic Kingdom. ver 1.00b",
                 use_rgb: bool = False,
                 shrink_ratio: float = 0.5,
                 clip_top: int = 26,
                 clip_bottom: int = 476,
                 clip_left: int = 32,
                 clip_right: int = 416) -> None:
        """
        初期化処理

        Parameters
        ----------
        grid_size : list, optional
            グリッドサイズ, by default [384, 450, 1]
        app_class_name : str, optional
            キャプチャ対象のウィンドウクラス名, by default BASE
        app_window_name : str, optional
            キャプチャ対象のウィンドウ名, by default 東方紺珠伝　～ Legacy of Lunatic Kingdom. ver 1.00b
        use_rgb : bool, optional
            RGB画像を使う, by default False
        shrink_ratio : float, optional
            画像縮小率, by default 0.5
        clip_top : int, optional
            画像を切り取る位置 (top), by default 26
        clip_bottom : int, optional
            画像を切り取る位置 (bottom), by default 476
        clip_left : int, optional
            画像を切り取る位置 (left), by default 32
        clip_right : int, optional
            画像を切り取る位置 (right), by default 416
        """
        super(Kanjuden, self).__init__()

        self.grid_size: np.ndarray = np.zeros(grid_size)
        self.app_class_name = app_class_name
        self.app_window_name = app_window_name
        self.use_rgb = use_rgb
        self.shrink_ratio = shrink_ratio
        self.clip_top = clip_top
        self.clip_bottom = clip_bottom
        self.clip_left = clip_left
        self.clip_right = clip_right
        self.action_space = spaces.Discrete(len(COMMAND_DICT))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.grid_size.shape, dtype=np.float
        )

        # ウィンドウ座標取得 (これ以降ウィンドウの場所を変えないこと)
        self.left, self.top, self.right, self.bottom = GetWindowRect(app_class_name, app_window_name)
        time.sleep(10)

        print("Title -> Mode Select")
        self.__press_and_release_key(SCAN_Z)

        print("Mode Select -> Rank Select")
        self.__press_and_release_key(SCAN_Z)

        print("Rank Select -> Player Select")
        self.__press_and_release_key(SCAN_Z)

        print("Player Select -> Game Start")
        self.__press_and_release_key(SCAN_Z)

        print("中断データから再開 (いいえ)")
        self.__press_and_release_key(SCAN_RIGHT)
        self.__press_and_release_key(SCAN_Z)
    
    def reset(self) -> np.ndarray:
        """
        環境のリセット

        Returns
        -------
        numpy.ndarray
            初期化後のグリッドのテンソル
        """
        self.grid_size: np.ndarray = np.zeros([0, 0, 0])
        return self.grid_size

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        環境の1ステップ実行

        Parameters
        ----------
        action : Any
            [description]

        Returns
        -------
        Tuple[Any, float, bool, Dict[str, Any]]
            [description]
        """
        self.__command_start(action)
        self.__command_end(action)  # TODO: 場所検討

        img: Image = ImageGrab.grab((self.left, self.top, self.right, self.bottom))
        imgArray: np.ndarray = np.asarray(img, dtype='uint8')
        if not self.use_rgb:
            imgArray = cv2.cvtColor(imgArray, cv2.COLOR_BGR2GRAY)
        imgArray = cv2.resize(imgArray, (int(imgArray.shape[1] * self.shrink_ratio), int(imgArray.shape[0] * self.shrink_ratio)))
        imgArray = imgArray[self.clip_top:self.clip_bottom, self.clip_left:self.clip_right]

        observation: np.ndarray = np.zeros(self.grid_size.shape())
        reward = 0
        done = False
        observation = imgArray.copy()
        self.current_screen = imgArray.copy()

        # テンプレートマッチングによる報酬計算
        # Chapter Finish
        cf_templates = [
            "template_chapter_finish_1.png",
            "template_chapter_finish_2.png",
            "template_chapter_finish_3.png",
            "template_chapter_finish_4.png",
            "template_chapter_finish_5.png"
        ]
        for cf_template in cf_templates:
            img_template = cv2helper.imread(f"./templates/{cf_template}", 0)
            result = cv2.matchTemplate(imgArray, img_template, cv2.TM_CCOEFF_NORMED)
            # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if len(np.where(result >= 0.95)[0]) > 0:
                print(f"Chapter Finished detected by {cf_template}")
                reward = 100
                break
            else:
                continue
        
        # Spell Card Bonus
        if reward == 0:
            img_template = cv2helper.imread("./templates/template_get_spell_card_bonus.png", 0)
            result = cv2.matchTemplate(imgArray, img_template, cv2.TM_CCOEFF_NORMED)
            if (len(np.where(result >= 0.95))[0]) > 0:
                print("Spell Card Bonus detected")
                reward = 500
        
        # 霊力
        # TODO: 前フレームより増えた数だけ +1

        # 得点
        # TODO: 前フレームより増えた数だけ +1

        # Mission Incomplete
        if reward == 0:
            img_template = cv2helper.imread("./templates/template_mission_incomplete.png", 0)
            result = cv2.matchTemplate(action, img_template, cv2.TM_CCOEFF_NORMED)
            if (len(np.where(result >= 0.95))[0]) > 0:
                print("Mission Incomplete detected")
                reward = -100
                done = True

        return observation, reward, done, {}

    def render(self, mode='console') -> None:
        """
        環境の描画

        Parameters
        ----------
        mode : str, optional
            [description], by default 'console'
        """
        if self.current_screen is None:
            raise ValueError("You must specify image source.")

        if mode == 'human':
            h, w = self.current_screen.shape
            # TODO: 霊力・得点の座標検出（速度次第…）
            root = tkinter.Tk()
            root.title("東方紺珠伝 - toho-ml-python")
            root.geometry(f"{w}x{h}")
            img_pil = Image.fromarray(self.current_screen)
            img_tk = ImageTk.PhotoImage(img_pil)
            canvas = tkinter.Canvas(root, width=w, height=h)
            canvas.pack()
            canvas.create_image(0, 0, image=img_tk, anchor='nw')
            root.mainloop()

        else:
            raise NotImplementedError(f"Not support render for mode {mode}.")
