import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

import environment.shanghai_alice_15th as env_module
from environment.flow import FlowConfig, FlowError, default_flow_config
from environment.screens import PLAYFIELD_RECT, POINT_P_TEMPLATE, Screen, TemplateLibrary
from utils import cv2helper
from utils.native import SCAN_ESC, SCAN_Z

FIXTURES = Path(__file__).parent / "fixtures" / "screens"
SHAPE = (450, 384)

SCREEN_FIXTURES = {
    Screen.TITLE_GAME_START: "title_game_start",
    Screen.MODE_SELECT: "mode_select",
    Screen.RANK_SELECT: "rank_select",
    Screen.PLAYER_SELECT: "player_select",
    Screen.RESUME_DIALOG: "resume_dialog",
    Screen.LOADING: "loading",
    Screen.PLAYING: "playing",
    Screen.GAME_OVER: "game_over",
    Screen.MANUAL_MENU: "manual_menu",
}

# 決定キー (Z) を押したときの画面遷移
TRANSITIONS = {
    Screen.TITLE_GAME_START: Screen.MODE_SELECT,
    Screen.MODE_SELECT: Screen.RANK_SELECT,
    Screen.RANK_SELECT: Screen.PLAYER_SELECT,
    Screen.PLAYER_SELECT: Screen.RESUME_DIALOG,
    Screen.RESUME_DIALOG: Screen.LOADING,
}


def load_screen(screen: Screen) -> np.ndarray:
    frame = cv2helper.imread(str(FIXTURES / f"{SCREEN_FIXTURES[screen]}.png"), 0)
    if screen is Screen.PLAYING:
        # 自機エリアに報酬用のアイコンが映り込まないよう、HUD だけ残して消す
        x0, y0, x1, y1 = PLAYFIELD_RECT
        frame = frame.copy()
        frame[y0:y1, x0:x1] = 0
    return frame


class FakeWindow:
    """GameWindow の代わりに、実機で撮った各画面を状態に応じて返す"""
    instance: Optional["FakeWindow"] = None
    key_events: List[Tuple[str, int]] = []
    initial_screen = Screen.PLAYING

    def __init__(self, app_class_name, app_window_name, spec) -> None:
        self.left, self.top, self.right, self.bottom = 0, 0, 960, 720
        self.spec = spec
        self.screen = FakeWindow.initial_screen
        self.gray = load_screen(self.screen)
        self.game_over_queued = False
        # 攻略失敗メニューのカーソル位置 (0 が「今回を無かった事にする」、2 が「操作説明」)。
        # 死亡の瞬間に押していた上キーで、先頭から最後へ動いた状況を再現するために、queue_game_over で動かせる
        self.leaked_cursor = 0
        # True なら、死亡の瞬間に Z も一緒にメニューに届き、動いたカーソルの項目が決定されてしまう
        self.leak_selects = False
        self.cursor = 0
        self.loading_grabs = 0
        self.grabs = 0
        # プレイ画面は動く (背景が流れる)。メニューなどは止まっている
        self.animate = self.screen is Screen.PLAYING
        FakeWindow.instance = self

    def show(self, screen: Screen) -> None:
        self.screen = screen
        self.gray = load_screen(screen)
        self.loading_grabs = 0
        self.animate = screen is Screen.PLAYING

    def show_manual_detail(self) -> None:
        """操作説明の詳細画面。HUD が残るので PLAYING と判定されるが、静止している"""
        self.screen = Screen.PLAYING
        self.gray = cv2helper.imread(str(FIXTURES / "manual_detail.png"), 0)
        self.animate = False

    def focus(self) -> bool:
        return True

    def grab(self) -> Tuple[np.ndarray, np.ndarray]:
        # ローディングは何回か取得すると終わってプレイ画面になる
        if self.screen is Screen.LOADING:
            self.loading_grabs += 1
            if self.loading_grabs > 4:
                self.show(Screen.PLAYING)
        gray = self.gray
        self.grabs += 1
        if self.animate:
            gray = gray.copy()
            gray[350:460, 40:160] = 120 * (self.grabs % 2)      # 背景が流れる代わりに、判定に影響しない場所を点滅させる
        x0, y0, x1, y1 = PLAYFIELD_RECT
        playfield = gray[y0:y1, x0:x1]
        # 本物の GameWindow と同じく、use_rgb なら観測は3チャンネル、判定用のグレーは常に2次元の全体
        observation = np.stack([playfield] * 3, axis=-1) if self.spec.use_rgb else playfield
        return observation, gray

    def on_keys(self, codes) -> None:
        if self.game_over_queued:
            # step() の入力の直後に画面がゲームオーバーになる状況を再現する
            self.game_over_queued = False
            self.cursor = self.leaked_cursor
            if self.leak_selects and self.leaked_cursor != 0 and SCAN_Z in codes:
                # 死亡の瞬間に、エージェントの上キーと Z が一緒にメニューに届き、攻略失敗を一度も見ないまま操作説明に入る
                self.show(Screen.MANUAL_MENU)
            else:
                self.show(Screen.GAME_OVER)
        elif self.screen is Screen.MANUAL_MENU and SCAN_ESC in codes:
            self.show(Screen.GAME_OVER)               # サブメニュー → 攻略失敗メニュー (カーソルは操作説明のまま)
            self.cursor = 2
        elif self.screen is Screen.PLAYING and not self.animate and SCAN_ESC in codes:
            self.show(Screen.MANUAL_MENU)             # 詳細画面 → サブメニュー
        elif self.screen is Screen.GAME_OVER and SCAN_ESC in codes:
            self.show(Screen.LOADING)                 # Esc は、カーソルに関わらず「今回を無かった事にする」
        elif self.screen is Screen.GAME_OVER and SCAN_Z in codes:
            if self.cursor == 0:
                self.show(Screen.LOADING)
            else:
                self.show(Screen.MANUAL_MENU)         # 先頭以外で決定すると、別の項目に入ってしまう
        elif SCAN_Z in codes and self.screen in TRANSITIONS:
            self.show(TRANSITIONS[self.screen])


@pytest.fixture
def keys(monkeypatch) -> List[Tuple[int, ...]]:
    """キー入力を記録し、時間待ちを省く"""
    pressed: List[Tuple[int, ...]] = []
    FakeWindow.key_events = []

    def hold_keys(codes, frames) -> None:
        pressed.append(tuple(codes))
        if FakeWindow.instance is not None:
            FakeWindow.instance.on_keys(codes)

    def press_key(code) -> None:
        # 押しっぱなしのキー (Z) の押し直し。ゲームには、押した瞬間だけキー入力として届く
        FakeWindow.key_events.append(("down", code))
        if FakeWindow.instance is not None:
            FakeWindow.instance.on_keys((code,))

    def release_key(code) -> None:
        FakeWindow.key_events.append(("up", code))

    FakeWindow.initial_screen = Screen.PLAYING
    FakeWindow.instance = None
    monkeypatch.setattr(env_module, "GameWindow", FakeWindow)
    monkeypatch.setattr(env_module, "HoldKeys", hold_keys)
    monkeypatch.setattr(env_module, "PressKey", press_key)
    monkeypatch.setattr(env_module, "ReleaseKey", release_key)
    return pressed


def make_env(**kwargs) -> env_module.Kanjuden:
    # 実時間を待たずに進める設定。画面ごとのキーは既定のまま
    flow = FlowConfig(keys=default_flow_config().keys, settle_sec=0.0, poll_interval_sec=0.0,
                      repress_interval_sec=0.05, stuck_timeout_sec=0.5, timeout_sec=5.0,
                      still_sec=0.3, motion_window_sec=0.3)
    return env_module.Kanjuden(flow_config=flow, **kwargs)


def queue_game_over() -> None:
    """次のキー入力の後にゲームオーバー画面になる"""
    FakeWindow.instance.game_over_queued = True


def test_Gymnasium_APIに従い_観測は観測空間に収まる(keys):
    env = make_env()

    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.shape == SHAPE
    assert env.observation_space.contains(observation)
    assert (reward, terminated, truncated) == (0, False, False)
    assert info["input_allowed"] is True


def test_RGB観測は3チャンネルで観測空間に収まり_ゲームオーバー判定も動く(keys):
    env = make_env(use_rgb=True)
    env.reset()
    queue_game_over()

    observation, reward, terminated, _, _ = env.step(0)

    assert env.observation_space.shape == (450, 384, 3)
    assert env.observation_space.contains(observation)
    assert (reward, terminated) == (-100, True)


def test_観測の大きさがずれていたら初期化で例外(keys):
    # 偽ウィンドウは縮小を適用せず (450, 384) を返すので、縮小率を指定すると観測空間とずれる
    with pytest.raises(ValueError, match="観測画像の大きさ"):
        make_env(shrink_ratio=0.5)


def test_初期化でタイトルからプレイ画面まで自動で進む(keys):
    FakeWindow.initial_screen = Screen.TITLE_GAME_START

    env = make_env()

    # Game Start → モード → 難易度 → キャラ → 中断データ確認、の5回
    assert keys == [(SCAN_Z,)] * 5
    assert FakeWindow.instance.screen is Screen.PLAYING
    assert env.observation_space.contains(env.reset()[0])


def test_すでにプレイ画面なら初期化でキーを押さない(keys):
    make_env()

    assert keys == []


def test_ゲームオーバーでterminatedになり_次のresetでリトライしてプレイ画面に戻る(keys):
    env = make_env()
    env.reset()
    keys.clear()

    queue_game_over()
    _, reward, terminated, _, info = env.step(0)
    assert (reward, terminated) == (-100, True)
    assert info["game_over_state"] == "hard"

    keys.clear()
    observation, _ = env.reset()

    assert keys == [(SCAN_ESC,)]
    assert FakeWindow.instance.screen is Screen.PLAYING
    assert env.observation_space.contains(observation)


def test_リトライ後は再びゲームオーバーを検出できる(keys):
    env = make_env()
    env.reset()
    for _ in range(2):
        queue_game_over()
        _, reward, terminated, _, _ = env.step(0)
        assert (reward, terminated) == (-100, True)
        env.reset()


def test_ゲームオーバーから進まなければresetはFlowError(keys, monkeypatch):
    env = make_env()
    env.reset()
    queue_game_over()
    env.step(0)
    # 決定キーを押しても画面が変わらない状況にする
    monkeypatch.setattr(env_module, "HoldKeys", lambda codes, frames: None)

    with pytest.raises(FlowError):
        env.reset()


def test_ゲームオーバー以外のresetはゲームに触れない(keys):
    env = make_env()
    env.reset()
    keys.clear()

    env.reset()

    assert keys == []


def test_アイテム報酬は増えた分だけで_毎フレーム加算されない(keys):
    env = make_env()
    env.reset()
    template = TemplateLibrary().get(POINT_P_TEMPLATE)
    x0, y0, _, _ = PLAYFIELD_RECT
    window = FakeWindow.instance
    window.gray = window.gray.copy()
    window.gray[y0 + 200:y0 + 200 + template.shape[0], x0 + 100:x0 + 100 + template.shape[1]] = template

    _, first, _, _, _ = env.step(0)
    _, second, _, _, _ = env.step(0)

    assert first > 0
    assert second == 0


# オーバーレイを検出する範囲 (environment/screens.py の OVERLAY_RULES と同じ)
CHAPTER_FINISH_RECT = (128, 173, 322, 307)
SPELL_CARD_BONUS_RECT = (92, 68, 338, 124)
STAGE_CLEAR_RECT = (148, 148, 312, 234)


def overlay_frame(fixture: str, *rects) -> np.ndarray:
    """アイコンなどの報酬要因が映り込まないよう、プレイ画面の HUD だけ残して、指定範囲だけ実機の画面を貼る"""
    frame = load_screen(Screen.PLAYING).copy()
    source = cv2helper.imread(str(FIXTURES / f"{fixture}.png"), 0)
    for x0, y0, x1, y1 in rects:
        frame[y0:y1, x0:x1] = source[y0:y1, x0:x1]
    return frame


def step_with(env: env_module.Kanjuden, frame: np.ndarray, times: int = 1):
    FakeWindow.instance.gray = frame
    return [env.step(0) for _ in range(times)]


def test_チャプタークリアは表示が始まった時に1回だけ報酬(keys):
    env = make_env()
    env.reset()

    results = step_with(env, overlay_frame("chapter_finish", CHAPTER_FINISH_RECT), 5)

    assert [r[1] for r in results] == [env_module.REWARD_CHAPTER_FINISH, 0, 0, 0, 0]
    assert results[0][4]["events"] == ["chapter_finish"]
    assert results[1][4]["events"] == []


def test_表示が十分途切れたら次のチャプタークリアにも報酬(keys):
    env = make_env()
    env.reset()
    chapter = overlay_frame("chapter_finish", CHAPTER_FINISH_RECT)
    none = load_screen(Screen.PLAYING)

    rewards = [r[1] for r in step_with(env, chapter) + step_with(env, none, 10) + step_with(env, chapter)]

    assert rewards[0] == env_module.REWARD_CHAPTER_FINISH
    assert rewards[-1] == env_module.REWARD_CHAPTER_FINISH
    assert sum(1 for r in rewards if r != 0) == 2


def test_スペルカードボーナスとチャプタークリアが同時に出たら両方の報酬(keys):
    env = make_env()
    env.reset()

    _, reward, terminated, _, info = step_with(
        env, overlay_frame("spell_card_bonus", CHAPTER_FINISH_RECT, SPELL_CARD_BONUS_RECT))[0]

    assert reward == env_module.REWARD_CHAPTER_FINISH + env_module.REWARD_SPELL_CARD_BONUS
    assert sorted(info["events"]) == ["chapter_finish", "spell_card_bonus"]
    assert terminated is False


def test_ステージクリアで終了し_次のresetは結果画面が終わるまで待つ(keys):
    env = make_env()
    env.reset()

    _, reward, terminated, _, info = step_with(env, overlay_frame("stage_clear", STAGE_CLEAR_RECT))[0]
    assert terminated is True
    assert env_module.REWARD_STAGE_CLEAR <= reward
    assert "stage_clear" in info["events"]

    # 結果画面が出たままでは、手動操作が必要な画面として止まる
    with pytest.raises(FlowError, match="手動"):
        env.reset()

    # 結果画面が終わってプレイ画面になれば、続けられる。次のステージクリアも数える
    FakeWindow.instance.show(Screen.PLAYING)
    observation, _ = env.reset()
    assert env.observation_space.contains(observation)
    _, reward, terminated, _, _ = step_with(env, overlay_frame("stage_clear", STAGE_CLEAR_RECT))[0]
    assert terminated is True


def test_ゲームオーバーの直前に上キーが届いてカーソルが動いても_Escなら確実にリトライする(keys):
    env = make_env()
    env.reset()
    FakeWindow.instance.leaked_cursor = 2          # 先頭から「操作説明」へ動いてしまった
    queue_game_over()
    env.step(0)
    keys.clear()

    observation, _ = env.reset()

    # 決定キーではなく Esc を押したので、操作説明に入らず、リトライされてプレイ画面に戻る
    assert keys == [(SCAN_ESC,)]
    assert FakeWindow.instance.screen is Screen.PLAYING
    assert env.observation_space.contains(observation)


def test_決定キーでは_カーソルが動いていると操作説明に入ってしまう(keys):
    # 以前の動作 (決定キー) の問題を再現する。Esc に変えた理由の記録
    env = make_env()
    env.reset()
    FakeWindow.instance.leaked_cursor = 2
    queue_game_over()
    env.step(0)

    env_module.HoldKeys([SCAN_Z], 4)       # 以前の動作と同じく、決定キーを送る

    assert FakeWindow.instance.screen is Screen.MANUAL_MENU


def test_死亡の瞬間のキーで操作説明に入っても_迷子として終了し_Escで戻ってリトライする(keys):
    # 実機で起きた問題の再現: 上キーと Z が攻略失敗メニューに届き、「操作説明」が選ばれる。環境は攻略失敗を一度も見ない
    env = make_env()
    env.reset()
    FakeWindow.instance.leaked_cursor = 2
    FakeWindow.instance.leak_selects = True
    queue_game_over()
    _, reward, terminated, _, info = env.step(2)          # 行動 2 = Z + 上
    assert FakeWindow.instance.screen is Screen.MANUAL_MENU and terminated is False

    steps = 1
    while not terminated and steps < 10:
        _, reward, terminated, _, info = env.step(0)
        steps += 1

    # サブメニューが続いたので、迷子として、ゲームオーバーと同じに扱って終了する
    assert terminated is True and info["lost"] is True and reward == env_module.REWARD_GAME_OVER
    assert info["events"] == ["game_over", "lost"]
    assert steps <= 6

    keys.clear()
    observation, _ = env.reset()

    # 決定キーではなく Esc で、サブメニュー → 攻略失敗メニュー → リトライ
    assert keys == [(SCAN_ESC,), (SCAN_ESC,)]
    assert FakeWindow.instance.screen is Screen.PLAYING
    assert env.observation_space.contains(observation)


def test_HUDが残る静止画面は_動きのなさで迷子とみなし_Escで戻る(keys):
    # 操作説明の詳細画面: HUD が残るので画面判定は PLAYING だが、画面は止まっている
    env = make_env()
    env.reset()
    FakeWindow.instance.show_manual_detail()

    # 上限は固定する (定数を変えたときに、テストが終わらなくならないように)
    results = [env.step(0) for _ in range(min(env_module.LOST_STILL_STEPS, 60) + 2)]

    terminated_at = next((i for i, r in enumerate(results) if r[2]), None)
    assert terminated_at is not None
    assert terminated_at >= env_module.LOST_STILL_STEPS - 1           # すぐには終了しない (静止が続いたとき)
    assert results[terminated_at][4]["lost"] is True

    keys.clear()
    observation, _ = env.reset()

    # 詳細画面 → サブメニュー → 攻略失敗メニュー → リトライ。入力は全て Esc
    assert keys == [(SCAN_ESC,)] * 3
    assert FakeWindow.instance.screen is Screen.PLAYING


def test_動くプレイ画面は_静止判定で終了しない(keys):
    env = make_env()
    env.reset()

    results = [env.step(0) for _ in range(min(env_module.LOST_STILL_STEPS, 60) * 2)]

    assert not any(r[2] for r in results)


def test_迷子で終了したときは_直近のステップの判定と動き量を保存する(keys, tmp_path):
    env = make_env(diagnostics_dir=str(tmp_path))
    env.reset()
    FakeWindow.instance.show_manual_detail()

    terminated, steps = False, 0
    while not terminated and steps < min(env_module.LOST_STILL_STEPS, 60) + 5:
        terminated = env.step(0)[2]
        steps += 1

    assert terminated is True
    dumped = [d for d in tmp_path.iterdir() if d.name.endswith("lost_step")]
    assert len(dumped) == 1
    rows = json.loads((dumped[0] / "frames.json").read_text(encoding="utf-8"))
    assert rows[-1]["still_steps"] >= env_module.LOST_STILL_STEPS
    assert {"screen", "motion", "action", "file"} <= set(rows[0])


def test_診断の保存先を指定しなければ_何も書かない(keys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = make_env()
    env.reset()
    FakeWindow.instance.show_manual_detail()
    for _ in range(min(env_module.LOST_STILL_STEPS, 60) + 2):
        env.step(0)
    env.dump_diagnostics("manual")

    assert list(tmp_path.iterdir()) == []


def z_events() -> List[str]:
    return [kind for kind, code in FakeWindow.key_events if code == SCAN_Z]


def test_Zは押しっぱなしで_周期の最後の1ステップだけ離して押し直す(keys):
    env = make_env(fire_period_steps=4)
    env.reset()
    FakeWindow.key_events.clear()

    per_step = []
    for _ in range(9):
        before = len(z_events())
        env.step(0)
        per_step.append(z_events()[before:])

    # 周期 4: 押す・保持・保持・離す、を繰り返す (会話は押し直しの辺がないと進まない)
    assert per_step == [["down"], [], [], ["up"], ["down"], [], [], ["up"], ["down"]]


def test_行動のキーにZは含まれず_方向とShiftだけ毎ステップ押して離す(keys):
    env = make_env()
    env.reset()
    keys.clear()

    for action in (0, 2, 10, 17):                  # 行動 2 = 上、10 = Shift + 左、17 = Shift + 下 + 左
        env.step(action)

    assert keys == [(), (env_module.SCAN_UP,), (env_module.SCAN_LSHIFT, env_module.SCAN_LEFT),
                    (env_module.SCAN_LSHIFT, env_module.SCAN_DOWN, env_module.SCAN_LEFT)]


def test_Zが押しっぱなしの間に死んでも_攻略失敗メニューは決定されず_Escで1回でリトライする(keys):
    # 実機で、死亡のたびに「操作説明」などが選ばれていた状況 (毎ステップ Z を押し直していた) の再現。
    # 押しっぱなしなら、死亡の瞬間に Z の押し直しの辺がないので、上キーでカーソルが動いても決定されない
    env = make_env()
    env.reset()
    env.step(0)                                    # 最初のステップで Z を押す
    FakeWindow.instance.leaked_cursor = 2
    FakeWindow.instance.leak_selects = True
    queue_game_over()

    _, reward, terminated, _, _ = env.step(2)      # 行動 2 = 上

    assert (reward, terminated) == (env_module.REWARD_GAME_OVER, True)
    assert FakeWindow.instance.screen is Screen.GAME_OVER
    keys.clear()
    env.reset()
    assert keys == [(SCAN_ESC,)]
    assert FakeWindow.instance.screen is Screen.PLAYING


def test_Zを押し直すステップで死ぬと_決定が届いて操作説明に入る(keys):
    # 押し直しの辺が残る周期の先頭 (最初のステップ) では、まだ漏れる。周期を長くするほど確率が下がる (残る課題の記録)
    env = make_env()
    env.reset()
    FakeWindow.instance.leaked_cursor = 2
    FakeWindow.instance.leak_selects = True
    queue_game_over()

    env.step(2)

    assert FakeWindow.instance.screen is Screen.MANUAL_MENU


def test_resetでZを離し_次のステップで押し直す(keys):
    env = make_env()
    env.reset()
    env.step(0)
    env.step(0)
    FakeWindow.key_events.clear()

    env.reset()
    assert ("up", SCAN_Z) in FakeWindow.key_events
    FakeWindow.key_events.clear()
    env.step(0)

    assert z_events() == ["down"]


def test_fire_period_stepsは2以上(keys):
    with pytest.raises(ValueError, match="fire_period_steps"):
        make_env(fire_period_steps=1)
