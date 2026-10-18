import json
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pytest

from environment.diagnostics import FrameRecorder
from environment.flow import FlowConfig, FlowError, GameFlow, default_flow_config
from environment.screens import Detection, Screen
from utils.native import SCAN_DOWN, SCAN_ESC, SCAN_Z

Z = (0x2C,)


class FakeGame:
    """
    画面・キー入力・時計の偽物

    sleep で時計が進む。キー入力のたびに transitions に従って画面が切り替わる。
    """
    def __init__(self, screen: Screen, transitions: Optional[Dict[Screen, Screen]] = None) -> None:
        self.screen = screen
        self.transitions = transitions or {}
        self.presses: List[Screen] = []
        self.codes: List[Sequence[int]] = []
        self.now = 0.0
        self.frames: List[Screen] = []
        # プレイ画面は動く (背景が流れる)。止めると、操作説明の詳細画面のように HUD が残ったまま静止する画面を再現できる
        self.moving = True
        self.grabs = 0

    def grab(self) -> np.ndarray:
        self.grabs += 1
        frame = np.zeros((480, 640), dtype=np.uint8)
        if self.moving:
            frame[100:300, 100:300] = 120 * (self.grabs % 2)
        return frame

    def detect(self, frame: np.ndarray) -> Detection:
        self.frames.append(self.screen)
        return Detection(self.screen, 0.0)

    def press(self, codes: Sequence[int]) -> None:
        self.presses.append(self.screen)
        self.codes.append(tuple(codes))
        self.screen = self.transitions.get(self.screen, self.screen)

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def clock(self) -> float:
        return self.now


def make_flow(game: FakeGame, keys: Dict[Screen, Sequence[int]], **overrides) -> GameFlow:
    class Detector:
        detect = staticmethod(game.detect)

    return GameFlow(game.grab, game.press, Detector(), FlowConfig(keys=keys, **overrides),
                    sleep=game.sleep, clock=game.clock)


def test_すでにプレイ画面なら何も押さない():
    game = FakeGame(Screen.PLAYING)

    make_flow(game, {Screen.GAME_OVER: Z}).advance_to_playing()

    assert game.presses == []


def test_ゲームオーバーから決定キーを1回押してプレイ画面まで進む():
    game = FakeGame(Screen.GAME_OVER, {Screen.GAME_OVER: Screen.LOADING})
    # ローディングは一定時間で自動的にプレイ画面になる
    original_sleep = game.sleep

    def sleep(seconds: float) -> None:
        original_sleep(seconds)
        if game.screen is Screen.LOADING and game.now > 5:
            game.screen = Screen.PLAYING

    game.sleep = sleep

    make_flow(game, {Screen.GAME_OVER: Z}).advance_to_playing()

    assert game.presses == [Screen.GAME_OVER]


def test_タイトルからプレイ画面まで画面ごとのキーで順に進む():
    game = FakeGame(Screen.TITLE_GAME_START, {
        Screen.TITLE_GAME_START: Screen.MODE_SELECT,
        Screen.MODE_SELECT: Screen.RANK_SELECT,
        Screen.RANK_SELECT: Screen.PLAYER_SELECT,
        Screen.PLAYER_SELECT: Screen.RESUME_DIALOG,
        Screen.RESUME_DIALOG: Screen.PLAYING,
    })

    make_flow(game, dict(default_flow_config().keys)).advance_to_playing()

    assert game.presses == [Screen.TITLE_GAME_START, Screen.MODE_SELECT, Screen.RANK_SELECT,
                            Screen.PLAYER_SELECT, Screen.RESUME_DIALOG]


def test_画面が出てから待ち時間を置いて入力する():
    game = FakeGame(Screen.GAME_OVER, {Screen.GAME_OVER: Screen.PLAYING})
    pressed_at: List[float] = []
    original_press = game.press
    game.press = lambda codes: (pressed_at.append(game.now), original_press(codes))[1]

    make_flow(game, {Screen.GAME_OVER: Z}, settle_sec=1.5).advance_to_playing()

    assert pressed_at[0] >= 1.5


def test_進まなければ再入力し_上限を超えたらFlowError():
    game = FakeGame(Screen.GAME_OVER)

    with pytest.raises(FlowError, match="進みませんでした"):
        make_flow(game, {Screen.GAME_OVER: Z}, max_presses_per_screen=2).advance_to_playing()

    assert len(game.presses) == 2


def test_再入力で進めば止まる():
    game = FakeGame(Screen.GAME_OVER)
    original_press = game.press

    def press(codes: Sequence[int]) -> None:
        original_press(codes)
        if len(game.presses) == 2:
            game.screen = Screen.PLAYING

    game.press = press

    make_flow(game, {Screen.GAME_OVER: Z}, max_presses_per_screen=3).advance_to_playing()

    assert len(game.presses) == 2


def test_キーの無い画面で動けなければ押さずにFlowError():
    # キャラ選択画面でモード・難易度が想定と違う場合など、手動操作が必要な画面
    game = FakeGame(Screen.PLAYER_SELECT_MISMATCH)

    with pytest.raises(FlowError, match="手動"):
        make_flow(game, dict(default_flow_config().keys), stuck_timeout_sec=5).advance_to_playing()

    assert game.presses == []


def test_判定できない画面が続いたらFlowError():
    game = FakeGame(Screen.UNKNOWN)

    with pytest.raises(FlowError):
        make_flow(game, dict(default_flow_config().keys), stuck_timeout_sec=5).advance_to_playing()

    assert game.presses == []


def test_時間切れはFlowError():
    game = FakeGame(Screen.LOADING)

    with pytest.raises(FlowError, match="プレイ画面になりませんでした"):
        make_flow(game, {}, stuck_timeout_sec=1000).advance_to_playing(timeout_sec=10)


def test_1フレームだけのプレイ画面判定では完了としない():
    game = FakeGame(Screen.GAME_OVER)
    calls = {"n": 0}

    def detect(frame: np.ndarray) -> Detection:
        calls["n"] += 1
        # 3回に1回だけ PLAYING と誤検出する
        return Detection(Screen.PLAYING if calls["n"] % 3 == 0 else Screen.GAME_OVER, 0.0)

    flow = GameFlow(game.grab, game.press, type("D", (), {"detect": staticmethod(detect)})(),
                    FlowConfig(keys={Screen.GAME_OVER: Z}, max_presses_per_screen=1),
                    sleep=game.sleep, clock=game.clock)

    with pytest.raises(FlowError):
        flow.advance_to_playing()


def test_既定の設定では_ゲームオーバーで決定キーではなくEscを押す():
    keys = default_flow_config().keys

    assert keys[Screen.GAME_OVER] == (SCAN_ESC,)
    assert SCAN_Z not in keys[Screen.GAME_OVER]
    # 操作説明に入ってしまった場合と、画面が止まったままの場合も、Esc で戻る
    escape_screens = {Screen.GAME_OVER, Screen.MANUAL_MENU, Screen.STUCK}
    assert all(keys[s] == (SCAN_ESC,) for s in escape_screens)
    # タイトルでカーソルが別の項目にあるときは、決定キーではなく下キーでカーソルを回す
    assert keys[Screen.TITLE] == (SCAN_DOWN,)
    # 他の画面は決定キー
    assert all(keys[s] == (SCAN_Z,) for s in keys if s not in escape_screens | {Screen.TITLE})


class BlinkingGame(FakeGame):
    """「Game Start」が点滅するタイトル: 検出できるのは周期の1回だけ (実機のログと同じ約0.67秒周期・1/4)"""
    def __init__(self, period_polls: int, start_polls: set) -> None:
        super().__init__(Screen.TITLE, {Screen.TITLE_GAME_START: Screen.PLAYING})
        self.period_polls = period_polls
        self.start_polls = start_polls
        self.polls = 0
        self.cursor_on_start = True

    def detect(self, frame):
        self.polls += 1
        bright = self.cursor_on_start and (self.polls % self.period_polls) in self.start_polls
        self.screen = Screen.TITLE_GAME_START if bright else Screen.TITLE
        if self.presses:
            self.screen = Screen.PLAYING
        return Detection(self.screen, 0.0)


def test_点滅する_Game_Start_は強調が一瞬でも選択中とみなして進む():
    # 実機のログ: 周期は約0.67秒、検出は1フレーム (0.17 秒) だけ。ここでは 4 回に 1 回だけ検出される
    game = BlinkingGame(period_polls=4, start_polls={1})
    game.press = lambda codes: game.presses.append(Screen.TITLE_GAME_START)

    make_flow(game, dict(default_flow_config().keys), poll_interval_sec=0.17).advance_to_playing()

    assert game.presses == [Screen.TITLE_GAME_START]


def test_強調が一度も検出されなければ_Quit_などを決定しない():
    game = BlinkingGame(period_polls=4, start_polls={1})
    game.cursor_on_start = False                 # カーソルは他の項目にある (点滅しない)
    game.press = lambda codes: game.codes.append(tuple(codes))      # 下キーではカーソルが動かない状況
    config = default_flow_config()

    with pytest.raises(FlowError, match="進みませんでした"):
        make_flow(game, dict(config.keys), poll_interval_sec=0.17, settle_by_screen=config.settle_by_screen,
                  max_presses_by_screen=config.max_presses_by_screen).advance_to_playing()

    assert game.codes and all(codes == (SCAN_DOWN,) for codes in game.codes)


def test_保持は指定した秒数だけ(monkeypatch):
    # 強調を一度検出したあと、保持時間を過ぎたら、タイトル (選択位置不明) に戻る
    game = FakeGame(Screen.TITLE_GAME_START)
    flow_calls = {"n": 0}

    def detect(frame):
        flow_calls["n"] += 1
        return Detection(Screen.TITLE_GAME_START if flow_calls["n"] == 1 else Screen.TITLE, 0.0)

    flow = GameFlow(game.grab, game.press, type("D", (), {"detect": staticmethod(detect)})(),
                    FlowConfig(keys=dict(default_flow_config().keys), poll_interval_sec=0.5, settle_sec=100.0,
                               stuck_timeout_sec=3.0, held={Screen.TITLE_GAME_START: (Screen.TITLE, 1.0)}),
                    sleep=game.sleep, clock=game.clock)

    with pytest.raises(FlowError, match="title"):
        flow.advance_to_playing(timeout_sec=60)


def test_実画像の点滅パターンでも_本物の判定器のまま決定キーを押せる():
    # 明るい瞬間 (1フレーム) と暗い瞬間 (3フレーム) を、実機のログと同じ周期で並べる
    from environment.screens import ScreenDetector
    from tests.test_screens import load_screen

    bright, dim = load_screen("title_game_start"), load_screen("title")
    game = FakeGame(Screen.TITLE)
    sequence = [dim, bright, dim, dim] * 10
    frames = iter(sequence)
    detector = ScreenDetector()
    assert [detector.detect(f).screen for f in (bright, dim)] == [Screen.TITLE_GAME_START, Screen.TITLE]

    def press(codes):
        game.presses.append(Screen.TITLE_GAME_START)
        raise StopIteration                              # 決定キーを押した時点で、検証を終える

    flow = GameFlow(lambda: next(frames), press, detector,
                    FlowConfig(keys=dict(default_flow_config().keys), poll_interval_sec=0.17),
                    sleep=game.sleep, clock=game.clock)

    with pytest.raises(StopIteration):
        flow.advance_to_playing()

    assert game.presses == [Screen.TITLE_GAME_START]
    assert game.now < 3.0                                # 数秒以内に押せる


class ManualGame(FakeGame):
    """操作説明の詳細画面 (HUD が残るので PLAYING と判定され、静止している) から、Esc で順に戻っていく"""
    def __init__(self, screen: Screen = Screen.PLAYING) -> None:
        super().__init__(screen)
        self.moving = False

    def press(self, codes: Sequence[int]) -> None:
        self.presses.append(self.screen)
        self.codes.append(tuple(codes))
        if self.screen in (Screen.PLAYING, Screen.UNKNOWN):
            self.screen = Screen.MANUAL_MENU          # 詳細画面 → サブメニュー
        elif self.screen is Screen.MANUAL_MENU:
            self.screen = Screen.GAME_OVER            # サブメニュー → 攻略失敗メニュー
        elif self.screen is Screen.GAME_OVER:
            self.screen = Screen.PLAYING              # Esc で「今回を無かった事にする」が選ばれ、リトライされる
            self.moving = True


@pytest.mark.parametrize("static_screen", [Screen.PLAYING, Screen.UNKNOWN])
def test_静止したままの画面は迷子とみなして_Escで攻略失敗メニューまで戻り_リトライする(static_screen):
    game = ManualGame(static_screen)

    make_flow(game, dict(default_flow_config().keys), still_sec=1.0).advance_to_playing()

    # 詳細画面 (静止) → サブメニュー → 攻略失敗メニュー → リトライ。入力は全て Esc
    assert game.presses == [static_screen, Screen.MANUAL_MENU, Screen.GAME_OVER]
    assert all(codes == (SCAN_ESC,) for codes in game.codes)


def test_動かないプレイ画面は_プレイ画面になったとはみなさない():
    # 判定が PLAYING でも、画面が動いていなければ、完了にしない (操作説明の詳細画面を、プレイ中と取り違えないため)
    game = ManualGame()

    with pytest.raises(FlowError, match="手動"):
        make_flow(game, {}, still_sec=1.0, stuck_timeout_sec=5).advance_to_playing()

    assert game.presses == []


def test_サブメニューは静止を待たずに検出して_Escで戻る():
    game = ManualGame(Screen.MANUAL_MENU)

    make_flow(game, dict(default_flow_config().keys), still_sec=100.0).advance_to_playing()

    assert game.presses == [Screen.MANUAL_MENU, Screen.GAME_OVER]


def test_動くプレイ画面は_静止判定の対象にならない():
    game = FakeGame(Screen.PLAYING)

    make_flow(game, dict(default_flow_config().keys), still_sec=0.2).advance_to_playing()

    assert game.presses == []


class FlickerGame(FakeGame):
    """
    攻略失敗メニューが出ているのに、判定だけが外れる状況の偽物

    入力するまでは、最初の数フレームだけ GAME_OVER と判定し、その後は PLAYING と判定する (メニューのフェードなどで
    マスクが外れた状況)。入力した後は after_press の画面を順に返し、尽きたら PLAYING を返す。
    """
    def __init__(self, visible_frames: int = 2, after_press: Sequence[Screen] = ()) -> None:
        super().__init__(Screen.GAME_OVER, {Screen.GAME_OVER: Screen.PLAYING})
        self.visible_frames = visible_frames
        self.after_press = list(after_press)

    def detect(self, frame: np.ndarray) -> Detection:
        self.frames.append(self.screen)
        if not self.presses:
            return Detection(Screen.GAME_OVER if len(self.frames) <= self.visible_frames else Screen.PLAYING, 0.0)
        return Detection(self.after_press.pop(0) if self.after_press else Screen.PLAYING, 0.0)


def test_攻略失敗を見たのに判定が外れても_入力せずにプレイ画面として抜けない():
    game = FlickerGame()

    make_flow(game, dict(default_flow_config().keys), answer_required=(Screen.GAME_OVER,)).advance_to_playing()

    assert game.presses == [Screen.GAME_OVER]
    assert game.codes == [(SCAN_ESC,)]


def test_answer_requiredが空なら_判定が外れた時点でプレイ画面として抜ける():
    # 上の挙動が answer_required によるものであることの確認
    game = FlickerGame()

    make_flow(game, dict(default_flow_config().keys)).advance_to_playing()

    assert game.presses == []


def test_入力した後の遷移中の判定は_元の画面とみなして再入力しない():
    # Esc の後、数秒間 GAME_OVER と UNKNOWN が混ざって判定されても、プレイ中に Esc (ポーズ) を押さない
    game = FlickerGame(after_press=[Screen.GAME_OVER] + [Screen.UNKNOWN] * 40)

    make_flow(game, dict(default_flow_config().keys), still_sec=100.0,
              answer_required=(Screen.GAME_OVER,)).advance_to_playing()

    assert game.presses == [Screen.GAME_OVER]


def test_保持時間が過ぎたら_攻略失敗のままとはみなさない():
    game = FlickerGame()

    make_flow(game, dict(default_flow_config().keys), settle_sec=2.0, answer_hold_sec=0.5,
              answer_required=(Screen.GAME_OVER,)).advance_to_playing()

    assert game.presses == []


def test_FlowErrorのときは直近のフレームを保存する(tmp_path):
    game = FakeGame(Screen.GAME_OVER)
    flow = make_flow(game, {Screen.GAME_OVER: Z}, max_presses_per_screen=1)
    flow._recorder = FrameRecorder(tmp_path)

    with pytest.raises(FlowError):
        flow.advance_to_playing()

    dumped = [d for d in tmp_path.iterdir() if d.name.endswith("flow_error")]
    assert len(dumped) == 1
    rows = json.loads((dumped[0] / "frames.json").read_text(encoding="utf-8"))
    assert rows and rows[0]["screen"] == "game_over" and (dumped[0] / rows[0]["file"]).exists()


def test_攻略失敗を経たリトライが終わったら_判定の経過を保存する(tmp_path):
    game = FlickerGame()
    flow = make_flow(game, dict(default_flow_config().keys), answer_required=(Screen.GAME_OVER,))
    flow._recorder = FrameRecorder(tmp_path)

    flow.advance_to_playing()

    dumped = [d for d in tmp_path.iterdir() if d.name.endswith("retry_done")]
    assert len(dumped) == 1
    rows = json.loads((dumped[0] / "frames.json").read_text(encoding="utf-8"))
    # 判定が外れた (detected=playing) のに、画面は game_over のままとして扱った経過が残る
    assert any(r["detected"] == "playing" and r["screen"] == "game_over" for r in rows)


class ScriptedGame(FakeGame):
    """画面の判定と、動いているかを、フレームごとの台本 [(判定, 動いているか), ...] で与える (尽きたら最後を繰り返す)"""
    def __init__(self, script: Sequence[tuple]) -> None:
        super().__init__(Screen.GAME_OVER)
        self.script = list(script)

    def _current(self) -> tuple:
        return self.script[min(self.grabs - 1, len(self.script) - 1)]

    def grab(self) -> np.ndarray:
        self.grabs += 1
        frame = np.zeros((480, 640), dtype=np.uint8)
        if self._current()[1]:
            frame[100:300, 100:300] = 120 * (self.grabs % 2)
        return frame

    def detect(self, frame: np.ndarray) -> Detection:
        self.frames.append(self.screen)
        return Detection(self._current()[0], 0.0)

    def press(self, codes: Sequence[int]) -> None:
        self.presses.append(self._current()[0])
        self.codes.append(tuple(codes))


def test_検出できないまま入力した後は_メニューの遷移中に動いて見えても_すぐにはプレイ画面として抜けない():
    # 実機の再現: 操作説明のサブメニューは判定が PLAYING のまま。攻略失敗の保持で Esc を押すと、攻略失敗メニューに戻る間の
    # 暗転 (約0.5秒) は、判定が PLAYING で、画面も動く。そこで抜けると、方策の最初の入力が攻略失敗メニューに届いて、
    # また「操作説明」に入る
    go, play = Screen.GAME_OVER, Screen.PLAYING
    game = ScriptedGame([(go, False)] * 2 + [(play, False)] * 10 + [(play, True)] * 3 + [(go, False)] * 10
                        + [(play, True)] * 40)
    flow = make_flow(game, dict(default_flow_config().keys), answer_required=(go,), play_confirm_sec=1.5)

    flow.advance_to_playing()

    assert len(game.codes) == 2      # 暗転の後の、検出できた攻略失敗に、もう一度 Esc を押した


def test_play_confirm_secが0なら_暗転の途中で抜ける():
    # 上の挙動が play_confirm_sec によるものであることの確認
    go, play = Screen.GAME_OVER, Screen.PLAYING
    game = ScriptedGame([(go, False)] * 2 + [(play, False)] * 10 + [(play, True)] * 3 + [(go, False)] * 10
                        + [(play, True)] * 40)

    make_flow(game, dict(default_flow_config().keys), answer_required=(go,)).advance_to_playing()

    assert len(game.codes) == 1
    assert game.grabs < 17


def test_検出できた攻略失敗に入力した後は_リトライの演出の途中でも待たずに抜ける():
    # 待つと、ゲームが再開してから方策が動くまで、自機が入力なしで止まったままになる
    go, play = Screen.GAME_OVER, Screen.PLAYING
    game = ScriptedGame([(go, False)] * 10 + [(play, True)] * 40)
    flow = make_flow(game, dict(default_flow_config().keys), answer_required=(go,), play_confirm_sec=1.5)

    flow.advance_to_playing()

    assert game.presses == [go]
    assert game.grabs <= 10 + 5


def test_攻略失敗を見ていない遷移には_動き続ける待ちを課さない():
    game = ScriptedGame([(Screen.LOADING, True)] * 2 + [(Screen.PLAYING, True)] * 10)

    make_flow(game, {}, answer_required=(Screen.GAME_OVER,), play_confirm_sec=100.0).advance_to_playing()

    assert game.grabs <= 6


def test_タイトルでカーソルが別の項目なら_決定キーではなく下キーで回して_Game_Startを探す():
    # 「ゲームを終了」の後はカーソルが Quit にある。決定キーを押すとゲームが終了してしまう
    game = FakeGame(Screen.TITLE, {Screen.TITLE: Screen.TITLE_GAME_START, Screen.TITLE_GAME_START: Screen.MODE_SELECT,
                                   Screen.MODE_SELECT: Screen.PLAYING})
    keys = dict(default_flow_config().keys)

    make_flow(game, keys, settle_by_screen=default_flow_config().settle_by_screen).advance_to_playing()

    assert game.presses == [Screen.TITLE, Screen.TITLE_GAME_START, Screen.MODE_SELECT]
    assert game.codes == [(SCAN_DOWN,), (SCAN_Z,), (SCAN_Z,)]


def test_タイトルで_Game_Startが見つからなくても_決定キーは押さずにFlowError():
    game = FakeGame(Screen.TITLE)
    config = default_flow_config()

    with pytest.raises(FlowError, match="進みませんでした"):
        make_flow(game, dict(config.keys), settle_by_screen=config.settle_by_screen,
                  max_presses_by_screen=config.max_presses_by_screen).advance_to_playing()

    assert game.codes and all(codes == (SCAN_DOWN,) for codes in game.codes)
    assert len(game.codes) == config.max_presses_by_screen[Screen.TITLE]


def test_タイトルの下キーは_待ち時間を置いてから押す():
    # 起動時に「Game Start」にカーソルがあっても、点滅の暗い瞬間に誤って下キーを押さないため
    game = FakeGame(Screen.TITLE, {Screen.TITLE: Screen.TITLE_GAME_START, Screen.TITLE_GAME_START: Screen.PLAYING})
    pressed_at: List[float] = []
    original_press = game.press
    game.press = lambda codes: (pressed_at.append(game.now), original_press(codes))[1]

    make_flow(game, dict(default_flow_config().keys),
              settle_by_screen=default_flow_config().settle_by_screen).advance_to_playing()

    assert pressed_at[0] >= 3.0


def test_攻略失敗の後にタイトルへ出たら_経過を保存する(tmp_path):
    game = ScriptedGame([(Screen.GAME_OVER, False)] * 10 + [(Screen.TITLE, False)] * 5
                        + [(Screen.TITLE_GAME_START, False)] * 3 + [(Screen.PLAYING, True)] * 40)
    config = default_flow_config()
    flow = make_flow(game, dict(config.keys), answer_required=config.answer_required,
                     settle_by_screen=config.settle_by_screen, play_confirm_sec=0.5)
    flow._recorder = FrameRecorder(tmp_path)

    flow.advance_to_playing()

    dumped = [d for d in tmp_path.iterdir() if d.name.endswith("title_after_game_over")]
    assert len(dumped) == 1
    rows = json.loads((dumped[0] / "frames.json").read_text(encoding="utf-8"))
    assert rows[0]["screen"] == "game_over" and rows[-1]["screen"] == "title"      # 保存はタイトルの最初のフレームを含む


def test_タイトルに戻った後の選択画面の遷移を_保持で攻略失敗とみなして_Escを押さない():
    # 実機のログ: 「ゲームを終了」でタイトルに戻った後、選択画面の遷移 (UNKNOWN) が「game_over」と表示された
    go, title, unknown = Screen.GAME_OVER, Screen.TITLE_GAME_START, Screen.UNKNOWN
    script = [(go, False)] * 2 + [(title, False)] * 2 + [(unknown, True)] * 12 + [(Screen.PLAYING, True)] * 10
    config = default_flow_config()

    game = ScriptedGame(script)
    make_flow(game, dict(config.keys), answer_required=config.answer_required,
              answer_cancel_screens=config.answer_cancel_screens).advance_to_playing()

    assert game.codes == []


def test_answer_cancel_screensが空なら_選択画面の遷移を攻略失敗とみなしてEscを押す():
    # 上の挙動が answer_cancel_screens によるものであることの確認
    go, title, unknown = Screen.GAME_OVER, Screen.TITLE_GAME_START, Screen.UNKNOWN
    script = [(go, False)] * 2 + [(title, False)] * 2 + [(unknown, True)] * 12 + [(Screen.PLAYING, True)] * 10
    config = default_flow_config()

    game = ScriptedGame(script)
    make_flow(game, dict(config.keys), answer_required=config.answer_required).advance_to_playing()

    assert game.codes == [(SCAN_ESC,)]


def test_リトライの遷移画面_Restartでは_何も押さずに終わるのを待つ():
    # 実機: Esc で「今回を無かった事にする」を選んだ後に、約 0.4 秒出る。HUD が残るので、以前は PLAYING と取り違えて、早く抜けていた
    config = default_flow_config()
    game = ScriptedGame([(Screen.RESTART, True)] * 4 + [(Screen.PLAYING, True)] * 10)

    make_flow(game, dict(config.keys), answer_required=config.answer_required).advance_to_playing()

    assert game.codes == []
    assert game.grabs >= 4 + 3                              # Restart の間は抜けず、PLAYING が 3 フレーム続いてから抜ける
