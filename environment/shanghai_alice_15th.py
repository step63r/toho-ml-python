import gymnasium as gym
import logging
import numpy as np
import time
import tkinter

from gymnasium import spaces
from pathlib import Path
from PIL import Image, ImageTk
from typing import Any, Callable, Dict, Optional, Tuple


from environment.capture import DEFAULT_APP_CLASS_NAME, DEFAULT_APP_WINDOW_NAME, FrameSpec, GameWindow
from environment.diagnostics import FrameRecorder
from environment.flow import FlowConfig, GameFlow, default_flow_config
from environment.screens import (
    POINT_P_TEMPLATE, POINT_TEN_TEMPLATE,
    STILL_DIFF, GameOverState, GameOverTracker, OnsetDetector, Screen, ScreenDetector, TemplateLibrary,
    frame_motion, playfield_signature)
from utils import cv2helper
from utils.native import (
    COMMAND_DICT, SCAN_Z, SCAN_RIGHT, SCAN_LEFT, SCAN_UP, SCAN_DOWN, SCAN_LSHIFT, HoldKeys, PressKey, ReleaseKey)


logger = logging.getLogger(__name__)

# 報酬。チャプタークリア・スペルカードボーナス・ステージクリアは、表示が始まった瞬間に1回だけ与える
REWARD_CHAPTER_FINISH = 100
REWARD_SPELL_CARD_BONUS = 500
REWARD_STAGE_CLEAR = 1000
REWARD_GAME_OVER = -100

# プレイ中に現れない画面。これらが続いたら、エージェントの入力でメニューなどに入ってしまった (迷子) とみなす。
# 例: 死亡の瞬間に押していたキーが攻略失敗メニューに届き、「操作説明」などが選ばれた場合は、
# 攻略失敗の画面を一度も見ないまま、環境が操作説明の中でキーを押し続けることになる
OFF_PLAY_SCREENS = frozenset({
    Screen.MANUAL_MENU, Screen.TITLE, Screen.TITLE_GAME_START, Screen.MODE_SELECT, Screen.RANK_SELECT,
    Screen.PLAYER_SELECT, Screen.PLAYER_SELECT_MISMATCH, Screen.RESUME_DIALOG})
OFF_PLAY_STEPS = 3
# 画面が止まったまま (操作説明の詳細画面のように HUD が残る静止画面) になったとみなすステップ数。
# 1ステップは約0.13秒なので、約2.7秒。ステージクリア等の動く画面は最小でも 6 の動きがあるため、誤検出しない
LOST_STILL_STEPS = 20
# Z (ショット) は押しっぱなしにして、この周期のうち 1 ステップだけ離す。
# 毎ステップ押し直すと、死亡の瞬間の押し直しが攻略失敗メニューに決定として届き、ほぼ毎回「操作説明」などが選ばれる。
# 押しっぱなしなら押し直しの辺がない。会話は Z を離して押し直さないと進まないので、周期的に離す
FIRE_PERIOD_STEPS = 10


class Kanjuden(gym.Env):
    """
    「東方紺珠伝」向けカスタムGymnasium環境

        - 目的：ゲームオーバーにならない
        - 行動：(8方向 + 無し) × Shift の 18 通り。Z (ショット) は環境が管理する (FIRE_PERIOD_STEPS)
        - 状態：画面イメージ
        - 報酬：チャプタークリア・スペルカードボーナス・ステージクリア・アイテム取得で加点、ゲームオーバーで減点
        - エピソード完了：ゲームオーバーまたはステージクリア (次の reset() でプレイ画面まで進める)
    """
    metadata = {"render_modes": ["human"]}

    def _release_all_keys(self) -> None:
        """環境で使用し得る全てのキーを強制的に離す"""
        for key in {SCAN_Z, SCAN_LEFT, SCAN_RIGHT, SCAN_UP, SCAN_DOWN, SCAN_LSHIFT}:
            ReleaseKey(key)
        self._fire_down = False
        self._fire_phase = 0
        self._held_moves = set()

    def _send_action(self, action: Any) -> None:
        """
        行動のキーを送る

        Z は、押しっぱなしにして、周期の最後のステップだけ離す。
        方向と Shift は、既定では 1 ステップだけ押して離す (action_hold_frames)。step_period_sec を指定した場合は、
        次のステップまで押しっぱなしにして、行動が変わったキーだけを押し直す (人の操作と同じ動き)。
        """
        move_keys = [code for code in COMMAND_DICT.get(int(action), []) if code != SCAN_Z]
        fire = self._fire_phase < self.fire_period_steps - 1
        if fire and not self._fire_down:
            PressKey(SCAN_Z)
        elif not fire and self._fire_down:
            ReleaseKey(SCAN_Z)
        self._fire_down = fire
        self._fire_phase = (self._fire_phase + 1) % self.fire_period_steps
        if self.step_period_sec is None:
            HoldKeys(move_keys, self.action_hold_frames)
            return
        wanted = set(move_keys)
        for code in self._held_moves - wanted:
            ReleaseKey(code)
        for code in wanted - self._held_moves:
            PressKey(code)
        self._held_moves = wanted

    def _wait_for_next_capture(self) -> None:
        """前の撮影から step_period_sec たつまで待ってから撮る (キーを押したままの時間を、毎ステップ同じにする)"""
        if self._last_capture_at is not None:
            remaining = self._last_capture_at + self.step_period_sec - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last_capture_at = self._clock()

    def __init__(self,
                 app_class_name: str = DEFAULT_APP_CLASS_NAME,
                 app_window_name: str = DEFAULT_APP_WINDOW_NAME,
                 use_rgb: bool = False,
                 shrink_ratio: float = 1.0,
                 clip_top: int = 16,
                 clip_bottom: int = 466,
                 clip_left: int = 32,
                 clip_right: int = 416,
                 action_hold_frames: int = 4,
                 fire_period_steps: int = FIRE_PERIOD_STEPS,
                 step_period_sec: Optional[float] = None,
                 flow_config: Optional[FlowConfig] = None,
                 debug_dump_path: Optional[str] = None,
                 diagnostics_dir: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 auto_advance: bool = True) -> None:
        """
        初期化処理

        Parameters
        ----------
        app_class_name : str, optional
            キャプチャ対象のウィンドウクラス名, by default BASE
        app_window_name : str, optional
            キャプチャ対象のウィンドウ名, by default 東方紺珠伝　～ Legacy of Lunatic Kingdom. ver 1.00b
        use_rgb : bool, optional
            RGB画像を観測に使う, by default False
        shrink_ratio : float, optional
            観測画像の縮小率 (画面判定用の画像には影響しない), by default 1.0
        clip_top : int, optional
            画像を切り取る位置 (top)。ゲーム本来の解像度 640x480 での座標, by default 16
        clip_bottom : int, optional
            画像を切り取る位置 (bottom), by default 466
        clip_left : int, optional
            画像を切り取る位置 (left), by default 32
        clip_right : int, optional
            画像を切り取る位置 (right), by default 416
        action_hold_frames : int, optional
            1ステップでキーを押し続けるフレーム数, by default 4
        fire_period_steps : int, optional
            Z (ショット) を、この周期のうち 1 ステップだけ離す。2 以上, by default 10
        step_period_sec : float, optional
            指定すると、1 ステップを固定の長さ (撮影の間隔) にして、方向・Shift を次のステップまで押しっぱなしにする。
            人のプレイの記録 (tools.record_demo) と同じ動きになる。省略すると、方向・Shift は毎ステップ
            action_hold_frames だけ押して離す, by default None
        flow_config : FlowConfig, optional
            画面遷移 (タイトル〜プレイ開始、ゲームオーバーからのリトライ) の設定, by default default_flow_config()
        debug_dump_path : str, optional
            指定すると毎ステップの観測画像をこのパスに書き出す, by default None
        diagnostics_dir : str, optional
            指定すると、直近のフレームと判定・動き量を溜めておき、迷子の検出やフローの失敗、
            dump_diagnostics() の呼び出しのときにこのディレクトリに書き出す, by default None
        render_mode : str, optional
            "human" のみ対応, by default None
        auto_advance : bool, optional
            False にすると、初期化でメニューを進めず、ゲームにキーを一切送らない (人のプレイの記録 tools.record_demo 用。
            この場合、observe() だけを使う), by default True
        """
        super(Kanjuden, self).__init__()

        self.use_rgb = use_rgb
        self.action_hold_frames = action_hold_frames
        if fire_period_steps < 2:
            raise ValueError("fire_period_steps は 2 以上にしてください (1 だと Z を一度も押さなくなります)。")
        self.fire_period_steps = fire_period_steps
        if step_period_sec is not None and step_period_sec <= 0:
            raise ValueError("step_period_sec は正の値にしてください。")
        self.step_period_sec = step_period_sec
        self._clock = clock
        self._sleep = sleep
        self._last_capture_at: Optional[float] = None
        self._fire_down = False
        self._fire_phase = 0
        self._held_moves: set = set()
        self.debug_dump_path = debug_dump_path
        self.render_mode = render_mode
        self.current_screen: Optional[np.ndarray] = None
        self._gray: Optional[np.ndarray] = None

        spec = FrameSpec(use_rgb=use_rgb, shrink_ratio=shrink_ratio, clip_top=clip_top,
                         clip_bottom=clip_bottom, clip_left=clip_left, clip_right=clip_right)
        shape = spec.observation_shape
        self.action_space = spaces.Discrete(len(COMMAND_DICT) - 1)
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        self._window = GameWindow(app_class_name, app_window_name, spec)
        self._templates = TemplateLibrary()
        self._detector = ScreenDetector(self._templates)
        self._game_over = GameOverTracker()
        self._flow_recorder = FrameRecorder(Path(diagnostics_dir)) if diagnostics_dir else None
        self._step_recorder = FrameRecorder(Path(diagnostics_dir), capacity=60) if diagnostics_dir else None
        self._flow = GameFlow(
            grab_gray=lambda: self._window.grab()[1],
            press=lambda codes: HoldKeys(codes, 4),
            detector=self._detector,
            config=flow_config or default_flow_config(),
            recorder=self._flow_recorder)
        self._chapter_finish = OnsetDetector()
        self._spell_card_bonus = OnsetDetector()
        self._stage_clear = OnsetDetector()
        self._needs_advance = False
        self._previous_signature: Optional[np.ndarray] = None
        self._still_steps = 0
        self._off_play_steps = 0
        self._previous_point_p_count = 0
        self._previous_point_ten_count = 0

        logger.info("app_class_name=%s app_window_name=%s use_rgb=%s", app_class_name, app_window_name, use_rgb)
        logger.info("observation shape=%s client rect=(%d, %d, %d, %d)", shape,
                    self._window.left, self._window.top, self._window.right, self._window.bottom)

        # 切り取り位置が画面の外にはみ出していると、学習が始まってから気付くことになる
        observation, _ = self._window.grab()
        if observation.shape != shape:
            raise ValueError(
                f"観測画像の大きさが想定と異なります: 実際 {observation.shape}, 想定 {shape}。"
                "shrink_ratio / clip_* を確認してください。")

        # 初回起動: プレイ画面まで自動遷移 (すでにプレイ画面なら何もしない)
        if auto_advance:
            self._window.focus()
            self._release_all_keys()
            self._flow.advance_to_playing()

    def _grab(self) -> np.ndarray:
        """画面を取得して観測画像を返す。画面判定用のグレー画像 (640x480 全体) は self._gray に保持する"""
        observation, gray = self._window.grab()
        self.current_screen = observation
        self._gray = gray
        if self.debug_dump_path:
            cv2helper.imwrite(self.debug_dump_path, observation)
        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        環境のリセット

        前のエピソードがゲームオーバーかステージクリアで終わっていた場合は、プレイ画面になるまで
        進めてから観測を返す。それ以外で呼ばれた場合は、ゲームには触れずに現在の画面を返す。

        Returns
        -------
        Tuple[numpy.ndarray, Dict[str, Any]]
            観測画像と info

        Raises
        ------
        FlowError
            プレイ画面まで進めなかった
        """
        super().reset(seed=seed)
        self._release_all_keys()
        if self._needs_advance:
            self._window.focus()
            self._flow.advance_to_playing()
            self._needs_advance = False
        self._game_over.reset()
        for onset in (self._chapter_finish, self._spell_card_bonus, self._stage_clear):
            onset.reset()
        self._previous_point_p_count = 0
        self._previous_point_ten_count = 0
        self._previous_signature = None
        self._still_steps = 0
        self._off_play_steps = 0
        observation = self._grab()
        # 固定周期のモードで、最初のステップも、次の撮影まで一周期分キーを押したままにするため
        self._last_capture_at = self._clock()
        return observation, {}

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        環境の1ステップ実行

        Parameters
        ----------
        action : Any
            COMMAND_DICT のキー (0〜17)

        Returns
        -------
        Tuple[numpy.ndarray, float, bool, bool, Dict[str, Any]]
            観測画像, 報酬, terminated, truncated, info
        """
        # ゲームオーバーのフェードイン中に入力すると、メニュー操作の取りこぼしや誤入力になる
        if self._game_over.state is GameOverState.SOFT:
            self._release_all_keys()
        else:
            self._send_action(action)
        if self.step_period_sec is not None:
            self._wait_for_next_capture()
        return self.observe(action)

    def observe(self, action: Optional[Any] = None) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        キーを送らずに、今の画面から観測・報酬・終了を求める (step() の、キー入力を除いた後半)

        人のプレイの記録 (tools.record_demo) が、学習と同じ報酬・終了の判定を使うためのもの。
        action は、診断用の記録に残すだけ (省略可)。
        """
        info: Dict[str, Any] = {}
        observation = self._grab()
        gray = self._gray
        detection = self._detector.detect(gray)
        # 報酬用のアイコンは、HUD側に同じ絵があるため自機エリアだけで照合する
        playfield = self._detector.playfield(gray)

        reward = 0
        terminated = False
        events = []

        if self._chapter_finish.update(Screen.CHAPTER_FINISH in detection.overlays):
            logger.info("Chapter Finish detected")
            reward += REWARD_CHAPTER_FINISH
            events.append("chapter_finish")

        if self._spell_card_bonus.update(Screen.SPELL_CARD_BONUS in detection.overlays):
            logger.info("Spell Card Bonus detected")
            reward += REWARD_SPELL_CARD_BONUS
            events.append("spell_card_bonus")

        if self._stage_clear.update(Screen.STAGE_CLEAR in detection.overlays):
            logger.info("Stage Clear detected")
            reward += REWARD_STAGE_CLEAR
            terminated = True
            self._needs_advance = True
            events.append("stage_clear")

        # 霊力・得点は、画面内のアイコン数が前回より増えた分だけ加点する
        if reward == 0:
            current = self._templates.count(playfield, POINT_P_TEMPLATE)
            reward = max(current - self._previous_point_p_count, 0)
            self._previous_point_p_count = current

        if reward == 0:
            current = self._templates.count(playfield, POINT_TEN_TEMPLATE)
            reward = max(current - self._previous_point_ten_count, 0)
            self._previous_point_ten_count = current

        if self._game_over.update(detection.game_over_score) is GameOverState.HARD:
            reward = REWARD_GAME_OVER
            terminated = True
            self._needs_advance = True
            events.append("game_over")
            info["game_over_confidence"] = detection.game_over_score

        # 攻略失敗のメニューを検出する前に、Z の押し直しがメニューを決定すると、メニューがほとんど見えないまま、
        # リトライの遷移画面 (Restart) になる (メニューが出始めてから 1 ステップ以内に消える)。Z を押しっぱなしにする設計で、
        # 保存したバッファの約 10% の死亡が、この形で見落とされていた。Restart は、リトライのたびに必ず現れるので、
        # 見落とした死亡として扱う。死亡した瞬間の数ステップ後になるが、被弾の演出 (リング) が映る状態が、終了の直前に並ぶ
        if detection.screen is Screen.RESTART and not terminated:
            reward = REWARD_GAME_OVER
            terminated = True
            self._needs_advance = True
            events.extend(["game_over", "missed_game_over"])
            info["missed_game_over"] = True
            logger.warning("攻略失敗を検出できないまま、リトライの遷移画面 (Restart) になりました。死亡として扱います。")

        # 攻略失敗を見逃したまま、メニューなどの中に入ってしまった場合 (迷子)。エピソードを終え、
        # 次の reset() で遷移制御が Esc で戻す。入ってしまうのは死亡の直後なので、ゲームオーバーとして扱う
        signature = playfield_signature(gray)
        motion = None if self._previous_signature is None else frame_motion(self._previous_signature, signature)
        if motion is not None and motion < STILL_DIFF:
            self._still_steps += 1
        else:
            self._still_steps = 0
        self._previous_signature = signature
        self._off_play_steps = self._off_play_steps + 1 if detection.screen in OFF_PLAY_SCREENS else 0
        if self._step_recorder:
            self._step_recorder.record(
                gray, action=-1 if action is None else int(action), screen=detection.screen.value, game_over_score=round(detection.game_over_score, 3),
                motion=None if motion is None else round(motion, 3), still_steps=self._still_steps,
                off_play_steps=self._off_play_steps)
        if (not terminated and detection.screen is not Screen.GAME_OVER
                and (self._still_steps >= LOST_STILL_STEPS or self._off_play_steps >= OFF_PLAY_STEPS)):
            logger.warning("プレイ画面から外れました (画面=%s, 静止 %d ステップ)。エピソードを終了します。",
                           detection.screen.value, self._still_steps)
            reward = REWARD_GAME_OVER
            terminated = True
            self._needs_advance = True
            self._still_steps = self._off_play_steps = 0
            events.extend(["game_over", "lost"])
            self.dump_diagnostics("lost")
            info["lost"] = True

        info["events"] = events
        # 判定した画面と攻略失敗の一致率。人のプレイの記録 (tools.record_demo) が、フレームごとに残す
        info["screen"] = detection.screen.value
        info["game_over_score"] = detection.game_over_score
        state = self._game_over.state
        info["game_over_state"] = state.value
        info["restart_ready"] = state is GameOverState.HARD
        info["input_allowed"] = state is GameOverState.NONE

        return observation, reward, terminated, False, info

    def dump_diagnostics(self, reason: str) -> None:
        """溜めてある直近のフレームを書き出す (diagnostics_dir を指定した場合のみ)"""
        for recorder in (self._flow_recorder, self._step_recorder):
            if recorder:
                recorder.dump(f"{reason}_{'flow' if recorder is self._flow_recorder else 'step'}")

    def render(self) -> None:
        """環境の描画 (render_mode="human" のみ)"""
        if self.current_screen is None:
            raise ValueError("You must specify image source.")

        if self.render_mode != 'human':
            return

        h, w = self.current_screen.shape[:2]
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

    def close(self) -> None:
        """押しっぱなしのキーが残らないようにする"""
        self._release_all_keys()
