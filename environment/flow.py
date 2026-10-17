import logging
import math
import numpy as np
import time

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from environment.screens import STILL_DIFF, Screen, ScreenDetector, frame_motion, playfield_signature
from utils.native import SCAN_ESC, SCAN_Z


logger = logging.getLogger(__name__)


class FlowError(RuntimeError):
    """画面遷移が期待どおりに進まなかった"""


@dataclass(frozen=True)
class FlowConfig:
    """画面遷移の設定 (秒数は実機で調整する)"""
    # 画面ごとに押すキー。ここに無い画面では何も押さずに待つ
    keys: Mapping[Screen, Sequence[int]] = field(default_factory=dict)
    poll_interval_sec: float = 0.1
    # 1フレームだけの誤検出・取りこぼしで判断しないための連続一致数
    stable_frames: int = 3
    # 画面が出てからの待ち時間。フェードイン完了前の入力は捨てられるため
    settle_sec: float = 0.7
    # 同じ画面で入力を繰り返す最短間隔。この間に画面が変わるかを確認する
    repress_interval_sec: float = 2.0
    max_presses_per_screen: int = 3
    # 同じ画面に居続けたら、進められないものとして諦める時間
    stuck_timeout_sec: float = 30.0
    timeout_sec: float = 300.0
    # 点滅して、検出が途切れる画面の保持: {保持する画面: (途切れたときに検出される画面, 保持する秒数)}。
    # タイトルで選択中の「Game Start」は約0.67秒周期で点滅し、明るい瞬間 (約1/4) しか検出できない。
    # 一度でも検出できたら、点滅の2周期以上は選択中とみなす。カーソルが他の項目にあるときは点滅しないので、
    # 一度も検出されず、保持もされない (誤って Quit などを決定しないための安全側の判断)
    held: Mapping[Screen, Tuple[Screen, float]] = field(
        default_factory=lambda: {Screen.TITLE_GAME_START: (Screen.TITLE, 1.5)})
    # 画面が動いているかの判定。プレイ画面 (PLAYING) になったとみなすには、画面が動いている必要がある。
    # 操作説明の詳細画面のように、HUD が残ったまま止まる画面は、画面判定ではプレイ画面と区別できないため。
    # 動き (フレーム間の差) が still_diff 未満の状態が still_sec 続いたら、迷子 (STUCK) とみなす
    still_diff: float = STILL_DIFF
    still_sec: float = 2.5
    motion_window_sec: float = 0.6
    # 画面ごとの、入力の上限回数 (省略した画面は max_presses_per_screen)
    max_presses_by_screen: Mapping[Screen, int] = field(default_factory=lambda: {Screen.STUCK: 6})


def default_flow_config() -> FlowConfig:
    """
    完全無欠モード・EASY・先頭のキャラで始め、ゲームオーバーでは「今回を無かった事にする」を選ぶ設定

    モード・難易度・「Game Start」以外の項目は、押す前に選択位置を確認できないものがある。
    モードと難易度はキャラ選択画面の下部表示 (PLAYER_SELECT / PLAYER_SELECT_MISMATCH) で、
    押した後に確認し、想定と違えばここで止める。
    中断データの確認ダイアログは、先頭の「はい」(続きから) が初期選択という前提で決定キーのみ押す。

    ゲームオーバーでは、決定キーではなく Esc を押す。攻略失敗メニューで Esc を押すと、カーソルの位置に関わらず
    「今回を無かった事にする」が選ばれる (ユーザー確認済み)。決定キーだと、死亡の瞬間に押していた上下キーで
    カーソルが動いていた場合に、「操作説明」などが選ばれてしまう。

    操作説明に入ってしまった場合は、Esc で戻る (サブメニュー → 攻略失敗メニュー → Esc でリトライ)。
    詳細画面は、HUD が残ったまま止まるので、動きのなさ (STUCK) で検出して、同じく Esc で戻る。
    """
    return FlowConfig(keys={
        Screen.TITLE_GAME_START: (SCAN_Z,),
        Screen.MODE_SELECT: (SCAN_Z,),
        Screen.RANK_SELECT: (SCAN_Z,),
        Screen.PLAYER_SELECT: (SCAN_Z,),
        Screen.RESUME_DIALOG: (SCAN_Z,),
        Screen.GAME_OVER: (SCAN_ESC,),
        Screen.MANUAL_MENU: (SCAN_ESC,),
        Screen.STUCK: (SCAN_ESC,),
    })


class GameFlow:
    """
    ゲームの画面遷移を、検出した画面に応じて入力しながら PLAYING になるまで進める

    固定時間のsleepだけで進めると、ロード時間のぶれや入力の取りこぼしに気付けず、
    想定外の画面で入力し続けてしまう。必ず画面の変化を確認し、進まなければ再入力し、
    それでも進まなければ例外にする。
    キー入力・画面取得・時計は外から渡し、実機なしでテストできるようにしている。
    """
    def __init__(self,
                 grab_gray: Callable[[], np.ndarray],
                 press: Callable[[Sequence[int]], None],
                 detector: ScreenDetector,
                 config: FlowConfig,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._grab_gray = grab_gray
        self._press = press
        self._detector = detector
        self._config = config
        self._sleep = sleep
        self._clock = clock

    def advance_to_playing(self, timeout_sec: Optional[float] = None) -> None:
        """
        現在の画面から、プレイ画面 (PLAYING) になるまで進める

        すでにプレイ画面なら何も押さずに戻る。

        Raises
        ------
        FlowError
            進められない画面で止まった、または時間切れ
        """
        cfg = self._config
        start = self._clock()
        deadline = start + (cfg.timeout_sec if timeout_sec is None else timeout_sec)

        current: Optional[Screen] = None
        stable = 0
        presses = 0
        entered_at = start
        last_press_at: Optional[float] = None
        last_seen: Dict[Screen, float] = {}
        previous_signature: Optional[np.ndarray] = None
        last_motion_at: Optional[float] = None

        while self._clock() < deadline:
            gray = self._grab_gray()
            screen = self._detector.detect(gray).screen
            now = self._clock()
            signature = playfield_signature(gray)
            if previous_signature is not None and frame_motion(previous_signature, signature) >= cfg.still_diff:
                last_motion_at = now
            previous_signature = signature
            moving = last_motion_at is not None and now - last_motion_at <= cfg.motion_window_sec
            still_for = now - (start if last_motion_at is None else last_motion_at)
            for held, (replaced, seconds) in cfg.held.items():
                if screen is held:
                    last_seen[held] = now
                elif screen is replaced and now - last_seen.get(held, -math.inf) <= seconds:
                    screen = held
            if screen in (Screen.PLAYING, Screen.UNKNOWN) and still_for >= cfg.still_sec:
                screen = Screen.STUCK
            if screen is current:
                stable += 1
            else:
                logger.info("画面: %s", screen.value)
                current, stable, presses, entered_at, last_press_at = screen, 1, 0, now, None

            if screen is Screen.PLAYING and stable >= cfg.stable_frames and moving:
                return

            keys = cfg.keys.get(screen)
            ready = stable >= cfg.stable_frames and now - entered_at >= cfg.settle_sec
            due = last_press_at is None or now - last_press_at >= cfg.repress_interval_sec
            if keys and ready and due:
                limit = cfg.max_presses_by_screen.get(screen, cfg.max_presses_per_screen)
                if presses >= limit:
                    raise FlowError(f"画面 {screen.value} で {presses} 回入力しても進みませんでした。")
                logger.info("入力 (%s, %d/%d)", screen.value, presses + 1, limit)
                self._press(keys)
                presses += 1
                last_press_at = self._clock()
            elif not keys and now - entered_at >= cfg.stuck_timeout_sec:
                raise FlowError(f"画面 {screen.value} から進められません (手動で操作が必要です)。")

            self._sleep(cfg.poll_interval_sec)

        raise FlowError(f"{self._clock() - start:.0f} 秒以内にプレイ画面になりませんでした (最後の画面: "
                        f"{current.value if current else 'なし'})。")
