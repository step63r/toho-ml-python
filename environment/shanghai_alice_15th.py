import cv2
import gym
import time
import tkinter
import numpy as np


from gym import spaces
from PIL import Image, ImageGrab, ImageTk
from typing import Any, Dict, Tuple


from utils import cv2helper
from utils.native import COMMAND_DICT, SCAN_ESC, SCAN_Z, SCAN_RIGHT, SCAN_LEFT, SCAN_UP, SCAN_DOWN, SCAN_LSHIFT, GetWindowRect, PressKey, ReleaseKey


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
        time.sleep(1)

    def _release_all_keys(self) -> None:
        """環境で使用し得る全てのキーを強制的に離す"""
        for key in {SCAN_Z, SCAN_LEFT, SCAN_RIGHT, SCAN_UP, SCAN_DOWN, SCAN_LSHIFT}:
            ReleaseKey(key)

    def __init__(self, grid_size=[450, 384],
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
            グリッドサイズ, by default [450, 384]
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
        self.action_space = spaces.Discrete(len(COMMAND_DICT) - 1)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.grid_size.shape, dtype=np.float
        )
        self.previous_point_p_count = 0
        self.previous_point_ten_count = 0
        # --- Mission Incomplete (Game Over) 検出用状態 ---
        # フェードインの早期段階での弱マッチを捕捉して入力を抑止する
        self._mission_incomplete_template = cv2helper.imread(
            "./environment/templates/template_mission_incomplete.png", 0
        )
        self._mi_soft_threshold = 0.80        # フェードイン初期用ソフト閾値
        self._mi_hard_threshold = 0.95        # 従来確定判定用ハード閾値
        self._mi_soft_required_frames = 5     # xz連続何フレームソフト閾値超で確定扱い
        self._mi_soft_counter = 0
        self._in_game_over_pending = False    # ソフト検出中 (入力抑止)
        self._in_game_over_cooldown = 0       # クールダウン残フレーム
        self.current_screen = None            # render安全用初期化

        # ウィンドウ座標取得 (これ以降ウィンドウの場所を変えないこと)
        self.left, self.top, self.right, self.bottom = GetWindowRect(app_class_name, app_window_name)

        print(f"==================================================")
        print(f"Properties")
        print(f"  grid_size      : {self.grid_size.shape}")
        print(f"  app_class_name : {self.app_class_name}")
        print(f"  app_window_name: {self.app_window_name}")
        print(f"  use_rgb        : {self.use_rgb}")
        print(f"  shrink_ratio   : {self.shrink_ratio}")
        print(f"  clip           : {self.clip_left}, {self.clip_top}, {self.clip_right}, {self.clip_bottom}")
        print(f"  rect           : {self.left}, {self.top}, {self.right}, {self.bottom}")
        print(f"==================================================")

        # 初回起動: ゲーム開始まで自動遷移
        self.fast_restart(initial=True)
    
    def reset(self) -> np.ndarray:
        """
        環境のリセット

        Returns
        -------
        numpy.ndarray
            初期化後のグリッドのテンソル
        """
        self.grid_size: np.ndarray = np.zeros([450, 384])
        # GameOver関連状態をリセット
        self._mi_soft_counter = 0
        self._in_game_over_pending = False
        self._in_game_over_cooldown = 0
        return self._capture_observation()

    # 観測キャプチャを共通化
    def _capture_observation(self) -> np.ndarray:
        img: Image = ImageGrab.grab((self.left, self.top, self.right, self.bottom), include_layered_windows=False, all_screens=True)
        imgArray: np.ndarray = np.asarray(img, dtype='uint8')
        if not self.use_rgb:
            imgArray = cv2.cvtColor(imgArray, cv2.COLOR_BGR2GRAY)
        imgArray = cv2.resize(imgArray, (int(imgArray.shape[1] * self.shrink_ratio), int(imgArray.shape[0] * self.shrink_ratio)))
        imgArray = imgArray[self.clip_top:self.clip_bottom, self.clip_left:self.clip_right]
        self.current_screen = imgArray.copy()
        cv2helper.imwrite('current_screen.png', imgArray)
        return imgArray

    def fast_restart(self, initial: bool = False) -> np.ndarray:
        """初回: メニューをゲーム開始まで進める / それ以降: ESCでリトライ画面確定 (想定)"""
        self._release_all_keys()
        if initial:
            sequence = [SCAN_Z, SCAN_Z, SCAN_Z, SCAN_Z, SCAN_RIGHT, SCAN_Z]
        else:
            sequence = [SCAN_ESC]
        for code in sequence:
            self.__press_and_release_key(code)
            time.sleep(2 * (1.0/60))
        # 状態初期化
        self._mi_soft_counter = 0
        self._in_game_over_pending = False
        self._in_game_over_cooldown = 0
        return self._capture_observation()

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
        info: Dict[str, Any] = {}
        suppress_input = self._in_game_over_pending  # クールダウンは学習側へ移譲
        if suppress_input:
            self._release_all_keys()
        else:
            self.__command_start(action)
            self.__command_end(action)

        imgArray = self._capture_observation()

        observation: np.ndarray = np.zeros(self.grid_size.shape)
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
            img_template = cv2helper.imread(f"./environment/templates/{cf_template}", 0)
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
            img_template = cv2helper.imread("./environment/templates/template_get_spell_card_bonus.png", 0)
            result = cv2.matchTemplate(imgArray, img_template, cv2.TM_CCOEFF_NORMED)
            if len(np.where(result >= 0.95)[0]) > 0:
                print("Spell Card Bonus detected")
                reward = 500
        
        # 霊力
        # 現在の画面と templates/point_p.png のテンプレートマッチングを行い、
        # 前回の画面と比較して増えていたらマッチした数だけ+1の報酬を与える
        if reward == 0:
            img_template = cv2helper.imread("./environment/templates/template_point_p.png", 0)
            result = cv2.matchTemplate(imgArray, img_template, cv2.TM_CCOEFF_NORMED)
            current_p_count = len(np.where(result >= 0.95)[0])
            if current_p_count > 0 and current_p_count > self.previous_point_p_count:
                print("Point +1 detected")
                reward = 1 * (current_p_count - self.previous_point_p_count)

        # 得点
        # 現在の画面と templates/point_ten.png のテンプレートマッチングを行い、
        # 前回の画面と比較して増えていたらマッチした数だけ+1の報酬を与える
        if reward == 0:
            img_template = cv2helper.imread("./environment/templates/template_point_ten.png", 0)
            result = cv2.matchTemplate(imgArray, img_template, cv2.TM_CCOEFF_NORMED)
            current_ten_count = len(np.where(result >= 0.95)[0])
            if current_ten_count > 0 and current_ten_count > self.previous_point_ten_count:
                print("Ten +1 detected")
                reward = 1 * (current_ten_count - self.previous_point_ten_count)

        # Mission Incomplete (Game Over) 検出 (フェードイン対応)
        if reward == 0:
            # 事前読込済テンプレートを使用
            result = cv2.matchTemplate(imgArray, self._mission_incomplete_template, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)

            if max_val >= self._mi_hard_threshold:
                # ハード閾値到達で確定
                print(f"Mission Incomplete HARD detected (val={max_val:.3f})")
                reward = -100
                done = True
                self._mi_soft_counter = 0
                self._in_game_over_pending = False
                info["game_over_state"] = "hard"
                info["game_over_confidence"] = max_val
                info["restart_ready"] = True
                info["input_allowed"] = False
            else:
                if max_val >= self._mi_soft_threshold:
                    # ソフト検出域
                    self._mi_soft_counter += 1
                    if not self._in_game_over_pending:
                        print(f"Mission Incomplete SOFT start (val={max_val:.3f}) -> suppress inputs")
                    self._in_game_over_pending = True
                    if self._mi_soft_counter >= self._mi_soft_required_frames:
                        # 規定フレーム継続で確定昇格
                        print(f"Mission Incomplete promoted from SOFT to HARD (counter={self._mi_soft_counter}, val={max_val:.3f})")
                        reward = -100
                        done = True
                        self._mi_soft_counter = 0
                        self._in_game_over_pending = False
                        info["game_over_state"] = "hard"
                        info["game_over_confidence"] = max_val
                        info["restart_ready"] = True
                        info["input_allowed"] = False
                else:
                    # 閾値未満: ソフト状態解除
                    if self._in_game_over_pending:
                        print(f"Mission Incomplete soft canceled (val={max_val:.3f})")
                    self._mi_soft_counter = 0
                    self._in_game_over_pending = False

        if "game_over_state" not in info:
            if self._in_game_over_pending:
                info["game_over_state"] = "soft"
                info["restart_ready"] = False
                info["input_allowed"] = False
            else:
                info["game_over_state"] = "none"
                info["restart_ready"] = False
                info["input_allowed"] = True

        return observation, reward, done, info

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
