import cv2
import logging
import numpy as np

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

from utils import cv2helper


logger = logging.getLogger(__name__)

# 実行ディレクトリに依存しないよう、モジュール位置から解決する
TEMPLATE_DIR = Path(__file__).parent / "templates"

# (x0, y0, x1, y1)。ゲーム本来の解像度 640x480 での座標
Rect = Tuple[int, int, int, int]

# 自機エリア。報酬用テンプレートはこの範囲で照合する (HUD側のアイコンを拾わないため)
PLAYFIELD_RECT: Rect = (32, 16, 416, 466)

POINT_P_TEMPLATE = "template_point_p.png"
POINT_TEN_TEMPLATE = "template_point_ten.png"

MATCH_THRESHOLD = 0.95

# マスクテンプレートの画素の種類
# 背景が違っても変わらない「白い文字」と「濃い縁取り」だけを比較対象にし、それ以外は無視する
MASK_WHITE_THRESHOLD = 235
MASK_DARK_THRESHOLD = 70
MASK_WHITE = 255
MASK_DARK = 0
MASK_IGNORE = 128


class Screen(Enum):
    """判定できる画面の種別"""
    UNKNOWN = "unknown"
    TITLE = "title"
    TITLE_GAME_START = "title_game_start"    # タイトルで Game Start にカーソルがある
    MODE_SELECT = "mode_select"
    RANK_SELECT = "rank_select"
    PLAYER_SELECT = "player_select"          # 完全無欠モード・EASY で選択済み
    PLAYER_SELECT_MISMATCH = "player_select_mismatch"    # モードか難易度が想定と違う
    RESUME_DIALOG = "resume_dialog"
    LOADING = "loading"
    # 攻略失敗メニューの「操作説明」の、9項目のサブメニュー。右側の HUD が残るので、PLAYING と区別が要る
    MANUAL_MENU = "manual_menu"
    PLAYING = "playing"
    # 画面判定器は返さない。遷移制御が「プレイ画面や不明な画面のまま画面が動かない」ことから推定する疑似画面
    # (操作説明の詳細画面など、HUD が残る静止画面)
    STUCK = "stuck"
    # 以下の3つはプレイ画面に重なって出るため、主画面とは別に overlays としても検出する
    CHAPTER_FINISH = "chapter_finish"
    SPELL_CARD_BONUS = "spell_card_bonus"
    STAGE_CLEAR = "stage_clear"
    GAME_OVER = "game_over"


class GameOverState(Enum):
    """ゲームオーバー画面の検出段階"""
    NONE = "none"
    SOFT = "soft"    # フェードイン途中。入力だけ止める
    HARD = "hard"    # 確定


def build_mask(crops: Sequence[np.ndarray], thin_only: int = 0) -> np.ndarray:
    """
    同じ画面を写した複数のグレー画像から、背景に依存しないマスクを作る

    全ての例で白い画素 (MASK_WHITE) と、その近傍 (2px 以内) にある全ての例で暗い画素 (MASK_DARK、
    文字の縁取り) だけを残し、それ以外は MASK_IGNORE にする。
    暗画素を近傍に限るのは、手元の例で偶然共通して暗かった背景まで拾うと、
    背景が明るい別の場面で一致率が下がる (過学習する) ため。

    thin_only に 0 より大きい値を渡すと、その大きさの核で開く (opening) 処理で残る太い白い塊 (背景の輪など) を
    白画素から除き、細い線の文字だけを残す。止まった背景の上に文字が載る画面で、背景まで覚えてしまうのを防ぐ。

    Parameters
    ----------
    crops : Sequence[numpy.ndarray]
        同じ大きさに切り出したグレー画像 (1枚以上)

    Returns
    -------
    numpy.ndarray
        MASK_WHITE / MASK_DARK / MASK_IGNORE のいずれかを持つ uint8 画像
    """
    white = np.logical_and.reduce([crop >= MASK_WHITE_THRESHOLD for crop in crops])
    if thin_only:
        kernel = np.ones((thin_only, thin_only), np.uint8)
        thick = cv2.dilate(cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, kernel), np.ones((3, 3), np.uint8))
        white &= ~thick.astype(bool)
    dark = np.logical_and.reduce([crop <= MASK_DARK_THRESHOLD for crop in crops])
    near_white = cv2.dilate(white.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    dark &= near_white
    mask = np.full(crops[0].shape, MASK_IGNORE, dtype=np.uint8)
    mask[white] = MASK_WHITE
    mask[dark] = MASK_DARK
    return mask


class MaskTemplate:
    """
    build_mask で作ったマスクとの一致率を求める

    一致率は「マスクの白画素が画像でも白い割合」と「暗画素が画像でも暗い割合」の平均 (0〜1)。
    画素数で重み付けした合算にしないのは、暗画素が多いマスクだと暗いだけの画面でも高い値が出るため。
    平均なら、暗いだけ・白いだけの画面は 0.5 止まりになる。
    最小値にしないのは、縁取りは背景が明るい場面で周囲と混ざって暗さが足りなくなりやすく、
    文字 (白画素) が合っていても背景違いで大きく下がってしまうため。
    """
    def __init__(self, mask: np.ndarray) -> None:
        self._white = (mask == MASK_WHITE).astype(np.float32)
        self._dark = (mask == MASK_DARK).astype(np.float32)
        self._white_count = float(self._white.sum())
        self._dark_count = float(self._dark.sum())
        if self._white_count == 0:
            raise ValueError("マスクに白画素 (文字) がありません。")

    @property
    def shape(self) -> Tuple[int, int]:
        return self._white.shape

    def best_score(self, frame: np.ndarray, search: Optional[Rect] = None) -> float:
        """search (省略時は画像全体) の範囲を探索したときの最大一致率"""
        if search is not None:
            x0, y0, x1, y1 = search
            frame = frame[y0:y1, x0:x1]
        rates = [cv2.matchTemplate((frame >= MASK_WHITE_THRESHOLD).astype(np.float32), self._white,
                                   cv2.TM_CCORR) / self._white_count]
        if self._dark_count:
            rates.append(cv2.matchTemplate((frame <= MASK_DARK_THRESHOLD).astype(np.float32), self._dark,
                                           cv2.TM_CCORR) / self._dark_count)
        return float(np.mean(rates, axis=0).max())


class TemplateLibrary:
    """
    テンプレート画像をキャッシュして照合する

    毎ステップのディスク読み込みを避けるのと、読み込み失敗を黙殺せず
    早期に例外にするために、cv2helper.imread を直接使わずここを経由する。
    """
    def __init__(self, template_dir: Path = TEMPLATE_DIR) -> None:
        self._template_dir = Path(template_dir)
        self._cache: Dict[str, np.ndarray] = {}
        self._masks: Dict[str, MaskTemplate] = {}

    def _read_gray(self, name: str) -> np.ndarray:
        image = cv2helper.imread(str(self._template_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"テンプレートを読み込めません: {self._template_dir / name}")
        return image

    def get(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = self._read_gray(name)
        return self._cache[name]

    def get_mask(self, name: str) -> MaskTemplate:
        if name not in self._masks:
            self._masks[name] = MaskTemplate(self._read_gray(name))
        return self._masks[name]

    def best_score(self, frame: np.ndarray, name: str) -> float:
        """フレーム内での最大一致度 (TM_CCOEFF_NORMED)"""
        result = cv2.matchTemplate(frame, self.get(name), cv2.TM_CCOEFF_NORMED)
        return float(result.max())

    def count(self, frame: np.ndarray, name: str, threshold: float = MATCH_THRESHOLD) -> int:
        """閾値以上で一致した位置の数"""
        result = cv2.matchTemplate(frame, self.get(name), cv2.TM_CCOEFF_NORMED)
        return int(np.count_nonzero(result >= threshold))


@dataclass(frozen=True)
class ScreenRule:
    """
    マスクによる画面判定のルール

    checks は (マスク名, 探索範囲) の組で、全てが threshold 以上のときに一致とする (最小値を一致率とする)。
    """
    screen: Screen
    checks: Tuple[Tuple[str, Rect], ...]
    threshold: float


_HEADER_SEARCH: Rect = (188, 0, 330, 64)
# キャラ選択画面の下部には、選んだモードと難易度が表示される。
# モード選択・難易度選択で意図しない項目を選んでしまった場合に、ここで気付くための確認用
_PLAYER_CONFIG = ("mask_player_config.png", (253, 418, 542, 480))

# 上から順に評価し、最初に一致した画面を返す。
# GAME_OVER と PLAYING は同じ HUD を持つため GAME_OVER を先に評価する。
# 選択画面は、より条件の多いもの (ダイアログ付き・選択内容の確認付き) を先に評価する。
# タイトルはカーソル位置まで分かる TITLE_GAME_START を TITLE より先に評価する。
# 閾値は、手元の撮影画像で「その画面の最低値」と「他の画面の最大値」の間に置いている
# (ゲームオーバー: 最低 0.89 (背景違いの1枚) / 他の画面の最大 0.56)。
GAME_OVER_RULE = ScreenRule(Screen.GAME_OVER, (("mask_game_over.png", PLAYFIELD_RECT),), 0.75)
SCREEN_RULES: Tuple[ScreenRule, ...] = (
    # 操作説明のサブメニューは HUD を持つので、PLAYING より先に評価する
    ScreenRule(Screen.MANUAL_MENU, (("mask_manual_menu.png", (58, 73, 312, 367)),), 0.85),
    ScreenRule(Screen.PLAYING, (("mask_playing.png", (420, 32, 502, 98)),), 0.85),
    ScreenRule(Screen.LOADING, (("mask_loading.png", (408, 360, 557, 424)),), 0.85),
    ScreenRule(Screen.RESUME_DIALOG, (("mask_resume_dialog.png", (284, 172, 568, 226)), _PLAYER_CONFIG), 0.85),
    ScreenRule(Screen.PLAYER_SELECT, (("mask_player_select.png", _HEADER_SEARCH), _PLAYER_CONFIG), 0.85),
    ScreenRule(Screen.PLAYER_SELECT_MISMATCH, (("mask_player_select.png", _HEADER_SEARCH),), 0.85),
    ScreenRule(Screen.RANK_SELECT, (("mask_rank_select.png", (200, 0, 330, 64)),), 0.85),
    ScreenRule(Screen.MODE_SELECT, (("mask_mode_select.png", (200, 0, 330, 64)),), 0.85),
    ScreenRule(Screen.TITLE_GAME_START, (("mask_title_game_start.png", (58, 220, 212, 274)),), 0.85),
    ScreenRule(Screen.TITLE, (("mask_title.png", (458, 28, 627, 412)),), 0.85),
)


# プレイ画面に重なって出る表示。主画面の判定とは独立に、全て評価する。
# Chapter Finish はボス戦でもスペルカードごとに出る。スペルカードボーナスやステージクリアの
# 背後にも出るので、同時に検出し得る。閾値は、検出した撮影画像の最低値 (0.87 / 0.98 / 1.0) と
# 検出しなかった画像の最大値 (0.71 / 0.67 / 0.5) の間に置いている。
OVERLAY_RULES: Tuple[ScreenRule, ...] = (
    ScreenRule(Screen.CHAPTER_FINISH, (("mask_chapter_finish.png", (128, 173, 322, 307)),), 0.80),
    ScreenRule(Screen.SPELL_CARD_BONUS, (("mask_spell_card_bonus.png", (92, 68, 338, 124)),), 0.80),
    ScreenRule(Screen.STAGE_CLEAR, (("mask_stage_clear.png", (148, 148, 312, 234)),), 0.80),
)
# 主画面として優先する順 (先に出るほど優先)。次の画面へ進む操作の対象にならない表示を先に置く
OVERLAY_PRIORITY = (Screen.STAGE_CLEAR, Screen.SPELL_CARD_BONUS, Screen.CHAPTER_FINISH)


# フレーム間の平均絶対差 (自機エリアを縮小した画像) がこれ未満なら、画面が止まっているとみなす。
# 手元の撮影では、止まった画面 (タイトル・中断データ確認・攻略失敗メニュー・ポーズ・操作説明) は 0.0〜0.6、
# 動く画面 (プレイ中・会話・ローディング・ステージクリア) は最小でも 6.1 だった
STILL_DIFF = 1.5
SIGNATURE_SIZE = (96, 112)


def playfield_signature(frame: np.ndarray) -> np.ndarray:
    """画面の動きを測るための、自機エリアの縮小画像 (縮小するのは、細かい雑音に反応しないため)"""
    x0, y0, x1, y1 = PLAYFIELD_RECT
    return cv2.resize(frame[y0:y1, x0:x1], SIGNATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def frame_motion(previous: np.ndarray, current: np.ndarray) -> float:
    """playfield_signature 同士の平均絶対差"""
    return float(np.abs(current - previous).mean())


@dataclass(frozen=True)
class Detection:
    screen: Screen
    # ゲームオーバーのフェードイン判定 (GameOverTracker) に使う
    game_over_score: float
    # 評価した画面ごとの一致率 (デバッグ表示用)
    scores: Dict[str, float] = field(default_factory=dict)
    # 検出したオーバーレイ (主画面が何であっても、重なっていれば全て入る)
    overlays: FrozenSet[Screen] = frozenset()


class ScreenDetector:
    """
    ゲームの画面種別を判定する

    入力は縮小・正規化済みの 640x480 のグレースケール画像全体 (切り取らない)。
    """
    def __init__(self, templates: Optional[TemplateLibrary] = None,
                 threshold: float = MATCH_THRESHOLD) -> None:
        self.templates = templates or TemplateLibrary()
        self.threshold = threshold

    @staticmethod
    def playfield(frame: np.ndarray) -> np.ndarray:
        """自機エリアだけを切り出す"""
        x0, y0, x1, y1 = PLAYFIELD_RECT
        return frame[y0:y1, x0:x1]

    def _score(self, frame: np.ndarray, rule: ScreenRule) -> float:
        return min(self.templates.get_mask(name).best_score(frame, search) for name, search in rule.checks)

    def detect(self, frame: np.ndarray) -> Detection:
        scores: Dict[str, float] = {}

        game_over_score = self._score(frame, GAME_OVER_RULE)
        scores[GAME_OVER_RULE.screen.value] = game_over_score
        if game_over_score >= GAME_OVER_RULE.threshold:
            return Detection(Screen.GAME_OVER, game_over_score, scores)

        overlays = set()
        for rule in OVERLAY_RULES:
            score = self._score(frame, rule)
            scores[rule.screen.value] = score
            if score >= rule.threshold:
                overlays.add(rule.screen)
        found = frozenset(overlays)

        for screen in OVERLAY_PRIORITY:
            if screen in found:
                return Detection(screen, game_over_score, scores, found)

        for rule in SCREEN_RULES:
            score = self._score(frame, rule)
            scores[rule.screen.value] = score
            if score >= rule.threshold:
                return Detection(rule.screen, game_over_score, scores, found)

        return Detection(Screen.UNKNOWN, game_over_score, scores, found)


class GameOverTracker:
    """
    ゲームオーバー画面のフェードインを追跡する

    フェードイン中は一致度が閾値に届かないため、緩い閾値 (soft) で入力を止め、
    soft が一定フレーム続いたら確定 (hard) とみなす。
    確定前にリトライ操作が走ると、暗転中の入力が捨てられて取りこぼすため。
    """
    def __init__(self, soft_threshold: float = 0.65, hard_threshold: float = GAME_OVER_RULE.threshold,
                 soft_required_frames: int = 5) -> None:
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.soft_required_frames = soft_required_frames
        self._soft_counter = 0
        self.state = GameOverState.NONE

    def reset(self) -> None:
        self._soft_counter = 0
        self.state = GameOverState.NONE

    def update(self, score: float) -> GameOverState:
        previous = self.state
        if score >= self.hard_threshold:
            self._soft_counter = 0
            self.state = GameOverState.HARD
        elif score >= self.soft_threshold:
            self._soft_counter += 1
            if self._soft_counter >= self.soft_required_frames:
                logger.info("Mission Incomplete: SOFT から HARD へ昇格 (score=%.3f)", score)
                self._soft_counter = 0
                self.state = GameOverState.HARD
            else:
                self.state = GameOverState.SOFT
        else:
            self._soft_counter = 0
            self.state = GameOverState.NONE

        if self.state != previous:
            logger.info("Mission Incomplete: %s -> %s (score=%.3f)", previous.value, self.state.value, score)
        return self.state


class OnsetDetector:
    """
    検出が始まった瞬間だけ True を返す

    Chapter Finish などは数秒間表示され続けるので、毎ステップ数えると報酬が何十回も入ってしまう。
    表示のフェード中に検出が一瞬途切れても二重に数えないよう、検出が連続 rearm_steps ステップ
    途切れてから、次の立ち上がりを受け付ける。
    時間ではなくステップ数にしているのは、ステップ間隔が環境の処理速度で変わっても挙動を決定的にするため。
    """
    def __init__(self, rearm_steps: int = 10) -> None:
        self.rearm_steps = rearm_steps
        self._armed = True
        self._inactive_steps = 0

    def reset(self) -> None:
        self._armed = True
        self._inactive_steps = 0

    def update(self, active: bool) -> bool:
        if active:
            self._inactive_steps = 0
            if self._armed:
                self._armed = False
                return True
            return False

        self._inactive_steps += 1
        if self._inactive_steps >= self.rearm_steps:
            self._armed = True
        return False
