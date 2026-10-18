import logging
import math
import numpy as np
import time

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from environment.diagnostics import FrameRecorder
from environment.screens import STILL_DIFF, Screen, ScreenDetector, frame_motion, playfield_signature
from utils.native import SCAN_DOWN, SCAN_ESC, SCAN_Z


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
    # 画面ごとの待ち時間 (省略した画面は settle_sec)
    settle_by_screen: Mapping[Screen, float] = field(default_factory=dict)
    # 一度検出したら、何か押すまでは出続けるはずの画面 (攻略失敗メニューなど)。入力しないのに PLAYING や UNKNOWN に
    # 判定が変わっても、answer_hold_sec の間は元の画面のままとみなして入力する。
    # メニューのフェードなどで判定が外れた隙に、入力せずにプレイ画面扱いで抜けると、方策の入力がメニューに入ってしまう
    answer_required: Tuple[Screen, ...] = ()
    answer_hold_sec: float = 5.0
    # answer_required の画面を、検出できないまま (answer_hold_sec の保持で) 入力した場合は、画面が途切れず動き続けて、
    # この秒数たってからでないとプレイ画面とみなさない。
    # メニューが開閉する遷移 (暗転など) は、判定が PLAYING で、画面も大きく動くが、0.5 秒ほどで止まるため、
    # 遷移の途中でプレイ画面とみなすと、方策の入力がメニューに入ってしまう。
    # 検出できた画面に入力したなら、遷移の先は確実なので待たない (待つと、ゲームが再開してから方策が動くまで、
    # 自機が入力なしで止まったままになる)
    play_confirm_sec: float = 0.0


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

    タイトルで「Game Start」が点滅しない (一度も検出されない) のは、カーソルが別の項目にあるとき。
    「ゲームを終了」で戻った後は、Quit にカーソルがある (ログと撮影で確認)。ここで決定キーを押すとゲームが終了するので、
    決定キーは押さず、下キーでカーソルを回して「Game Start」が点滅する (検出される) のを待つ。
    点滅は約0.67秒周期で、検出できる瞬間は約1/4なので、起動時に「Game Start」にカーソルがある場合に誤って下キーを
    押さないよう、3秒待ってから押す。
    """
    return FlowConfig(answer_required=(Screen.GAME_OVER,), play_confirm_sec=1.5,
                      settle_by_screen={Screen.TITLE: 3.0}, max_presses_by_screen={Screen.STUCK: 6, Screen.TITLE: 10},
                      keys={
        Screen.TITLE: (SCAN_DOWN,),
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
                 clock: Callable[[], float] = time.monotonic,
                 recorder: Optional[FrameRecorder] = None) -> None:
        self._grab_gray = grab_gray
        self._press = press
        self._detector = detector
        self._config = config
        self._sleep = sleep
        self._clock = clock
        self._recorder = recorder

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
        moving_since: Optional[float] = None
        # 見たのに、まだ入力していない画面 -> 最後に見た時刻。一度入力した画面は、入力で消えたものとして二度と入れない
        # (入力後の遷移中の判定を、元の画面とみなして再入力すると、プレイ中に Esc を押してポーズにしてしまう)
        awaiting: Dict[Screen, float] = {}
        answered = set()
        saw_game_over = False
        needs_confirm = False
        detected_run = 0
        previous_detected: Optional[Screen] = None
        titled = False
        dump_reason: Optional[str] = None

        def fail(message: str) -> FlowError:
            if self._recorder:
                self._recorder.dump("flow_error")
            return FlowError(message)

        while self._clock() < deadline:
            gray = self._grab_gray()
            screen = self._detector.detect(gray).screen
            detected = screen
            detected_run = detected_run + 1 if detected is previous_detected else 1
            previous_detected = detected
            now = self._clock()
            signature = playfield_signature(gray)
            motion = None if previous_signature is None else frame_motion(previous_signature, signature)
            if motion is not None and motion >= cfg.still_diff:
                last_motion_at = now
                if moving_since is None:
                    moving_since = now
            elif motion is not None:
                moving_since = None
            previous_signature = signature
            moving = last_motion_at is not None and now - last_motion_at <= cfg.motion_window_sec
            still_for = now - (start if last_motion_at is None else last_motion_at)
            for held, (replaced, seconds) in cfg.held.items():
                if screen is held:
                    last_seen[held] = now
                elif screen is replaced and now - last_seen.get(held, -math.inf) <= seconds:
                    screen = held
            if screen in cfg.answer_required:
                if screen not in answered:
                    awaiting[screen] = now
                saw_game_over = saw_game_over or screen is Screen.GAME_OVER
            elif screen in (Screen.PLAYING, Screen.UNKNOWN):
                for required, seen_at in awaiting.items():
                    if now - seen_at <= cfg.answer_hold_sec:
                        screen = required
                        break
            if screen in (Screen.PLAYING, Screen.UNKNOWN) and still_for >= cfg.still_sec:
                screen = Screen.STUCK
            if screen is current:
                stable += 1
            else:
                logger.info("画面: %s", screen.value)
                if saw_game_over and screen in (Screen.TITLE, Screen.TITLE_GAME_START) and not titled:
                    # 攻略失敗の後にタイトルへ出るのは、「ゲームを終了」が選ばれた場合。選ばれた経緯を残す
                    titled = True
                    dump_reason = "title_after_game_over"
                current, stable, presses, entered_at, last_press_at = screen, 1, 0, now, None

            if self._recorder:
                self._recorder.record(gray, screen=screen.value, detected=detected.value,
                                      motion=None if motion is None else round(motion, 3), moving=moving,
                                      stable=stable, presses=presses)
                if dump_reason:
                    self._recorder.dump(dump_reason)
            dump_reason = None
            confirmed = (not needs_confirm
                         or (moving_since is not None and now - moving_since >= cfg.play_confirm_sec))
            if screen is Screen.PLAYING and stable >= cfg.stable_frames and moving and confirmed:
                if self._recorder and saw_game_over:
                    self._recorder.dump("retry_done")
                return

            keys = cfg.keys.get(screen)
            settle = cfg.settle_by_screen.get(screen, cfg.settle_sec)
            ready = stable >= cfg.stable_frames and now - entered_at >= settle
            due = last_press_at is None or now - last_press_at >= cfg.repress_interval_sec
            if keys and ready and due:
                limit = cfg.max_presses_by_screen.get(screen, cfg.max_presses_per_screen)
                if presses >= limit:
                    raise fail(f"画面 {screen.value} で {presses} 回入力しても進みませんでした。")
                logger.info("入力 (%s, %d/%d)", screen.value, presses + 1, limit)
                self._press(keys)
                if screen in cfg.answer_required:
                    # 直近に、実際に検出できた画面への入力かどうか (保持で補った画面への入力は、先が確実でない)
                    needs_confirm = not (detected is screen and detected_run >= cfg.stable_frames)
                awaiting.pop(screen, None)
                answered.add(screen)
                presses += 1
                last_press_at = self._clock()
            elif not keys and now - entered_at >= cfg.stuck_timeout_sec:
                raise fail(f"画面 {screen.value} から進められません (手動で操作が必要です)。")

            self._sleep(cfg.poll_interval_sec)

        raise fail(f"{self._clock() - start:.0f} 秒以内にプレイ画面になりませんでした (最後の画面: "
                   f"{current.value if current else 'なし'})。")
