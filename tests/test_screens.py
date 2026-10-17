from pathlib import Path

import numpy as np
import pytest

from PIL import Image

from environment.capture import FrameSpec, preprocess
from environment.screens import (
    MASK_DARK, MASK_IGNORE, MASK_WHITE, PLAYFIELD_RECT, GameOverState, GameOverTracker, MaskTemplate, OnsetDetector,
    Screen, ScreenDetector, TemplateLibrary, build_mask)
from utils import cv2helper

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def templates() -> TemplateLibrary:
    return TemplateLibrary()


@pytest.fixture(scope="module")
def detector(templates) -> ScreenDetector:
    return ScreenDetector(templates)


def load_screen(name: str) -> np.ndarray:
    """実機で撮った画面を、判定と同じ 640x480 のグレー画像にしたもの"""
    return cv2helper.imread(str(FIXTURES / "screens" / f"{name}.png"), 0)


@pytest.mark.parametrize("name, expected", [
    ("title", Screen.TITLE),
    ("title_game_start", Screen.TITLE_GAME_START),
    ("mode_select", Screen.MODE_SELECT),
    ("rank_select", Screen.RANK_SELECT),
    ("player_select", Screen.PLAYER_SELECT),
    ("resume_dialog", Screen.RESUME_DIALOG),
    ("loading", Screen.LOADING),
    ("playing", Screen.PLAYING),
    ("game_over", Screen.GAME_OVER),
    ("game_over_boss", Screen.GAME_OVER),
    ("boss_battle", Screen.PLAYING),
    ("dialogue", Screen.PLAYING),
    ("chapter_finish", Screen.CHAPTER_FINISH),
    ("spell_card_bonus", Screen.SPELL_CARD_BONUS),
    ("stage_clear", Screen.STAGE_CLEAR),
    # ステージ2 (暗い背景)。マスクの作成に使っていない別の場面
    ("stage2_ready", Screen.PLAYING),
    ("stage2_play", Screen.PLAYING),
    ("stage2_chapter_finish", Screen.CHAPTER_FINISH),
    # ステージ開始直前の切り替わり。HUD が崩れていて判定できない (待つだけでよい)
    ("stage_transition", Screen.UNKNOWN),
    # ゲーム開始から2面序盤までの通し撮影から。マスクの作成には使っていない
    ("game_over_stage1", Screen.GAME_OVER),
    ("stage_clear_stage1", Screen.STAGE_CLEAR),
    ("resume_dialog_run", Screen.RESUME_DIALOG),
    ("loading_run", Screen.LOADING),
    # 会話中はプレイ画面と同じ HUD で、区別していない (会話中は決定キーを断続的に押す必要がある)
    ("dialogue_stage1", Screen.PLAYING),
    # 操作説明: サブメニューは HUD が残るのでマスクで区別する。詳細画面は 9 ページ分の文字が違うので区別せず、
    # HUD が残ったまま静止していること (動きのなさ) で検出する
    ("manual_menu", Screen.MANUAL_MENU),
    ("manual_detail", Screen.PLAYING),
])
def test_実機の画面を判定できる(detector, name, expected):
    assert detector.detect(load_screen(name)).screen is expected


def test_キャラ選択でモードか難易度が違えば区別する(detector):
    # 下部の「完全無欠モード ニュームーン級」を消して、別の選択をした状態にする
    frame = load_screen("player_select").copy()
    frame[418:480, 253:542] = 0

    assert detector.detect(frame).screen is Screen.PLAYER_SELECT_MISMATCH


def test_プレイ画面とゲームオーバー画面は同じHUDだがゲームオーバーを優先する(detector):
    playing = detector.detect(load_screen("playing"))
    game_over = detector.detect(load_screen("game_over"))

    assert playing.screen is Screen.PLAYING and playing.game_over_score < 0.7
    assert game_over.screen is Screen.GAME_OVER and game_over.game_over_score >= 0.85


def test_別の背景のゲームオーバー画面も判定できる(detector):
    # マスクの作成には使っていない、別の場面のクライアント領域画像 (960x720)
    rgb_client = np.asarray(Image.open(FIXTURES / "game_over_960x720.png").convert("RGB"))
    _, gray = preprocess(rgb_client, FrameSpec())

    detection = detector.detect(gray)

    assert detection.screen is Screen.GAME_OVER


def test_旧形式の自機エリア画像でも判定できる(detector):
    # 雲の背景で撮った旧フレーム。自機エリアだけなので、全体画像の同じ位置に置く
    x0, y0, x1, y1 = PLAYFIELD_RECT
    frame = np.zeros((480, 640), dtype=np.uint8)
    frame[y0:y1, x0:x1] = cv2helper.imread(str(FIXTURES / "game_over.png"), 0)

    assert detector.detect(frame).screen is Screen.GAME_OVER


@pytest.mark.parametrize("name", ["title", "mode_select", "rank_select", "player_select", "loading", "playing"])
def test_ゲームオーバーと誤認しない(detector, name):
    assert detector.detect(load_screen(name)).game_over_score < 0.7


def test_真っ黒な画面はUNKNOWN(detector):
    # 暗画素だけが一致して「暗い画面なら何でも一致」にならないことの確認
    assert detector.detect(np.zeros((480, 640), dtype=np.uint8)).screen is Screen.UNKNOWN


def test_真っ白な画面もUNKNOWN(detector):
    assert detector.detect(np.full((480, 640), 255, dtype=np.uint8)).screen is Screen.UNKNOWN


def test_オーバーレイは主画面と独立に全て検出する(detector):
    # スペルカードボーナスとステージクリアの背後には、Chapter Finish も同時に出ている
    assert detector.detect(load_screen("spell_card_bonus")).overlays == {Screen.SPELL_CARD_BONUS, Screen.CHAPTER_FINISH}
    assert detector.detect(load_screen("stage_clear")).overlays == {Screen.STAGE_CLEAR, Screen.CHAPTER_FINISH}
    assert detector.detect(load_screen("chapter_finish")).overlays == {Screen.CHAPTER_FINISH}


@pytest.mark.parametrize("name", ["title", "mode_select", "player_select", "loading", "playing", "boss_battle",
                                  "dialogue", "game_over", "game_over_boss"])
def test_オーバーレイが無い画面では検出しない(detector, name):
    assert detector.detect(load_screen(name)).overlays == frozenset()


def test_立ち上がり検出_続いている間は1回だけ():
    onset = OnsetDetector(rearm_steps=3)

    assert [onset.update(True) for _ in range(5)] == [True, False, False, False, False]


def test_立ち上がり検出_一瞬の途切れでは二重に数えない():
    onset = OnsetDetector(rearm_steps=3)

    results = [onset.update(a) for a in (True, True, False, False, True, True)]

    assert results == [True, False, False, False, False, False]


def test_立ち上がり検出_十分途切れたら次のイベントを数える():
    onset = OnsetDetector(rearm_steps=3)

    results = [onset.update(a) for a in (True, False, False, False, True)]

    assert results == [True, False, False, False, True]


def test_立ち上がり検出_resetで再び受け付ける():
    onset = OnsetDetector(rearm_steps=100)
    onset.update(True)

    onset.reset()

    assert onset.update(True) is True


def test_存在しないテンプレートは例外(tmp_path):
    with pytest.raises(FileNotFoundError):
        TemplateLibrary(tmp_path).get("none.png")


def test_マスク作成_全ての例で共通する白と暗だけを残す():
    a = np.array([[255, 255, 0, 100]], dtype=np.uint8)
    b = np.array([[255, 100, 0, 255]], dtype=np.uint8)

    mask = build_mask([a, b])

    assert mask.tolist() == [[MASK_WHITE, MASK_IGNORE, MASK_DARK, MASK_IGNORE]]


def test_マスク照合_白画素が無いマスクは作れない():
    with pytest.raises(ValueError):
        MaskTemplate(np.full((4, 4), MASK_DARK, dtype=np.uint8))


def test_マスク照合_位置が変わっても見つかり_暗いだけの画像は一致しない():
    template = MaskTemplate(build_mask([np.array([[0, 255, 255, 0]] * 3, dtype=np.uint8)]))
    frame = np.zeros((20, 30), dtype=np.uint8)
    frame[10:13, 7:11] = np.array([[0, 255, 255, 0]] * 3, dtype=np.uint8)

    assert template.best_score(frame) == pytest.approx(1.0)
    # 暗画素だけが一致する (白画素が一致しない) 画像は、平均で 0.5 止まり
    assert template.best_score(np.zeros((20, 30), dtype=np.uint8)) == pytest.approx(0.5)


def test_ゲームオーバー検出_ハード閾値で即確定():
    tracker = GameOverTracker()

    assert tracker.update(0.97) is GameOverState.HARD


def test_ゲームオーバー検出_ソフトが続くと確定に昇格():
    tracker = GameOverTracker(soft_required_frames=3)

    states = [tracker.update(0.70) for _ in range(3)]

    assert states == [GameOverState.SOFT, GameOverState.SOFT, GameOverState.HARD]


def test_ゲームオーバー検出_ソフトが途切れたら解除():
    tracker = GameOverTracker(soft_required_frames=3)

    tracker.update(0.70)
    tracker.update(0.70)
    assert tracker.update(0.30) is GameOverState.NONE
    # カウンタも戻っているので、再度ソフトに入っても直ちには確定しない
    assert tracker.update(0.70) is GameOverState.SOFT


@pytest.mark.parametrize("width, height", [(640, 480), (960, 720), (1280, 960)])
def test_前処理_ウィンドウの表示倍率によらず同じ大きさになる(width, height):
    rgb_client = np.zeros((height, width, 3), dtype=np.uint8)

    observation, gray = preprocess(rgb_client, FrameSpec())

    assert observation.shape == (450, 384)
    # 画面判定用のグレーは切り取らない全体
    assert gray.shape == (480, 640)


def test_前処理_RGB観測と観測の縮小():
    rgb_client = np.zeros((720, 960, 3), dtype=np.uint8)

    observation, gray = preprocess(rgb_client, FrameSpec(use_rgb=True, shrink_ratio=0.5))

    assert observation.shape == (225, 192, 3) == FrameSpec(use_rgb=True, shrink_ratio=0.5).observation_shape
    # 観測を縮小しても、画面判定用のグレーは等倍のまま
    assert gray.shape == (480, 640)


def test_前処理_グレー化は既存テンプレートに合わせてBGR2GRAY():
    # 純赤は BGR2GRAY (R を青として扱う) だと 255 * 0.114 ≒ 29、RGB2GRAY だと 255 * 0.299 ≒ 76 になる。
    # テンプレートの作成元と変換がずれて一致度が落ちるのを防ぐための固定
    rgb_client = np.zeros((720, 960, 3), dtype=np.uint8)
    rgb_client[..., 0] = 255

    _, gray = preprocess(rgb_client, FrameSpec())

    assert int(gray[0, 0]) == 29


def test_動きの測定_静止した画面とプレイ中の画面を分ける():
    from environment.screens import STILL_DIFF, frame_motion, playfield_signature

    still = playfield_signature(load_screen("manual_detail"))
    play_a, play_b = playfield_signature(load_screen("stage2_play")), playfield_signature(load_screen("boss_battle"))

    assert frame_motion(still, still) == 0.0 < STILL_DIFF
    # 別のプレイ画面同士は、大きく違う (実機のプレイ中の連続フレームは、最小でも約 6 の差があった)
    assert frame_motion(play_a, play_b) > STILL_DIFF * 3
