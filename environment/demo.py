"""
人のプレイの記録 (デモ) の、保存・読み込み・行動の決定

記録の単位は「フレーム」。環境と同じ周期 (既定 0.133 秒) で撮った画面と、その時点の判定を持つ。
各フレームの action は、そのフレームの撮影から次の撮影までの間に、人が押していた方向・Shift の組み合わせ
(環境の 18 通りの行動のどれか)。Z (ショット) は、環境が管理するので、行動に含めない (z_fraction に残すだけ)。
遷移 (画面, 行動, 報酬, 終了, 次の画面) は、隣り合うフレームから作る。報酬と終了は、そのフレームが「届いた」時点の値
(前の行動の結果) なので、遷移 k の報酬は frame[k + 1] の reward。
"""
import json
import os
import threading
import time

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from environment.screens import Screen
from utils.native import COMMAND_DICT, SCAN_DOWN, SCAN_LEFT, SCAN_LSHIFT, SCAN_RIGHT, SCAN_UP, SCAN_Z

# Windows の仮想キーコード
VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN = 0x25, 0x26, 0x27, 0x28
VK_SHIFT, VK_Z, VK_X, VK_ESC = 0x10, 0x5A, 0x58, 0x1B

# 判定した画面の名前を、番号にして保存する (並びを変えると、保存済みの記録が読めなくなるので、末尾にだけ足す)
SCREEN_NAMES: Tuple[str, ...] = tuple(screen.value for screen in Screen)

# フレームごとの出来事 (flags の各ビット)
FLAG_CHAPTER_FINISH = 1
FLAG_SPELL_CARD_BONUS = 2
FLAG_STAGE_CLEAR = 4
FLAG_GAME_OVER = 8
FLAG_LOST = 16
EVENT_FLAGS = {"chapter_finish": FLAG_CHAPTER_FINISH, "spell_card_bonus": FLAG_SPELL_CARD_BONUS,
               "stage_clear": FLAG_STAGE_CLEAR, "game_over": FLAG_GAME_OVER, "lost": FLAG_LOST}

# 押しているキー (方向・Shift の組み合わせ) の、環境の行動番号。環境の COMMAND_DICT と一致する (テストで確認)
_DIRECTION_INDEX = {(0, 0): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3, (0, 1): 4,
                    (-1, -1): 5, (1, -1): 6, (1, 1): 7, (-1, 1): 8}
SHIFT_OFFSET = 9


def keys_to_action(left: bool, right: bool, up: bool, down: bool, shift: bool) -> int:
    """押している方向キー・Shift を、環境の行動番号 (0〜17) にする。左右・上下を同時に押したら、打ち消し合う"""
    return _DIRECTION_INDEX[(int(right) - int(left), int(down) - int(up))] + (SHIFT_OFFSET if shift else 0)


def action_to_keys(action: int) -> Tuple[bool, bool, bool, bool, bool]:
    """環境の行動番号から、(左, 右, 上, 下, Shift) を求める (keys_to_action の逆。テストと、再生の確認用)"""
    codes = COMMAND_DICT[action]
    return (SCAN_LEFT in codes, SCAN_RIGHT in codes, SCAN_UP in codes, SCAN_DOWN in codes, SCAN_LSHIFT in codes)


# キーの状態: (左, 右, 上, 下, Shift, Z, X, Esc)
KeyState = Tuple[bool, bool, bool, bool, bool, bool, bool, bool]


def read_keys() -> KeyState:
    """今のキーの状態を読む (GetAsyncKeyState。ゲームが前面でも、他のウィンドウが前面でも読める)"""
    from utils.native import IsKeyDown
    return tuple(IsKeyDown(vk) for vk in (VK_LEFT, VK_RIGHT, VK_UP, VK_DOWN, VK_SHIFT, VK_Z, VK_X, VK_ESC))  # type: ignore


class KeySampler:
    """
    キーの状態を、短い間隔で読み続ける。撮影 (約 50 ms かかる) の間も、人の操作を取りこぼさないため、別のスレッドで動かす
    """
    def __init__(self, read: Callable[[], KeyState] = read_keys, clock: Callable[[], float] = time.perf_counter,
                 interval: float = 0.004) -> None:
        self._read = read
        self._clock = clock
        self._interval = interval
        self._samples: List[Tuple[float, KeyState]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def sample(self) -> None:
        state = self._read()
        with self._lock:
            self._samples.append((self._clock(), state))

    def start(self) -> None:
        def run() -> None:
            while not self._stop.is_set():
                self.sample()
                time.sleep(self._interval)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def consume(self, start: float, end: float) -> List[Tuple[float, KeyState]]:
        """
        [start, end) のサンプルを返す。区間の頭には、直前の状態を引き継ぐ (押しっぱなしのキーを、区間の最初から数えるため)。
        end 以降のサンプルは次の区間のために残し、区間の最後のサンプルも、次の区間の「直前の状態」として 1 件残す。
        """
        with self._lock:
            carry: Optional[Tuple[float, KeyState]] = None
            inside: List[Tuple[float, KeyState]] = []
            later: List[Tuple[float, KeyState]] = []
            for sample in self._samples:
                if sample[0] < start:
                    carry = sample
                elif sample[0] < end:
                    inside.append(sample)
                else:
                    later.append(sample)
            last = inside[-1] if inside else carry
            self._samples = ([last] if last else []) + later
        head = [(start, carry[1])] if carry and (not inside or inside[0][0] > start) else []
        return head + inside


def summarize_interval(samples: Sequence[Tuple[float, KeyState]], start: float, end: float) -> Dict[str, Any]:
    """
    区間 [start, end) のキーの状態から、行動と、参考の値を求める

    action: 方向・Shift の組み合わせのうち、押していた時間が最も長いもの (同じなら、後のもの)。サンプルが無ければ -1。
    z_fraction: Z を押していた時間の割合。x_pressed / esc_pressed: X (ボム)・Esc が一度でも押されたか。
    """
    if not samples or end <= start:
        return {"action": -1, "z_fraction": 0.0, "x_pressed": False, "esc_pressed": False}
    durations: Counter = Counter()
    z_time = 0.0
    x_pressed = esc_pressed = False
    for index, (t, state) in enumerate(samples):
        until = samples[index + 1][0] if index + 1 < len(samples) else end
        span = max(min(until, end) - max(t, start), 0.0)
        durations[keys_to_action(*state[:5])] += span
        z_time += span if state[5] else 0.0
        x_pressed = x_pressed or state[6]
        esc_pressed = esc_pressed or state[7]
    best = max(durations.values())
    action = [a for a, d in durations.items() if d == best][-1]
    return {"action": int(action), "z_fraction": z_time / (end - start), "x_pressed": bool(x_pressed),
            "esc_pressed": bool(esc_pressed)}


class DemoWriter:
    """
    フレームを溜めて、一定数ごとに part_XXXX.npz に書き出す

    最後に追加したフレームは、次の撮影までの行動が決まっていないので、次のフレームが来る (または close) まで書かない。
    書き込み途中で止まっても、書き終えた part は壊れないよう、別名に書いてから置き換える。
    """
    FIELDS = ("obs", "action", "t", "screen", "reward", "terminated", "flags", "game_over_score", "z_fraction",
              "x_pressed", "esc_pressed")

    def __init__(self, session_dir: Path, flush_every: int = 500, meta: Optional[Dict[str, Any]] = None,
                 background: bool = True) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.meta = dict(meta or {})
        self._background = background
        self._rows: List[Dict[str, Any]] = []
        self._part = 0
        self._frames = 0
        self._writers: List[threading.Thread] = []

    @property
    def frames(self) -> int:
        return self._frames

    def add(self, obs: np.ndarray, t: float, screen: str, reward: float, terminated: bool, flags: int,
            game_over_score: float) -> None:
        """新しいフレームを追加する (行動は、次のフレームの直前に set_action で決める)"""
        self._rows.append({"obs": obs, "action": -1, "t": t, "screen": SCREEN_NAMES.index(screen), "reward": reward,
                           "terminated": terminated, "flags": flags, "game_over_score": game_over_score,
                           "z_fraction": 0.0, "x_pressed": False, "esc_pressed": False})
        self._frames += 1

    def set_action(self, action: int, z_fraction: float, x_pressed: bool, esc_pressed: bool) -> None:
        """直前に追加したフレームの、次の撮影までの行動を決める"""
        row = self._rows[-1]
        row.update(action=action, z_fraction=z_fraction, x_pressed=x_pressed, esc_pressed=esc_pressed)
        if len(self._rows) > self.flush_every:
            self._write(self._rows[:self.flush_every])
            self._rows = self._rows[self.flush_every:]

    def _write(self, rows: List[Dict[str, Any]]) -> None:
        arrays = {
            "obs": np.stack([r["obs"] for r in rows]),
            "action": np.array([r["action"] for r in rows], dtype=np.int8),
            "t": np.array([r["t"] for r in rows], dtype=np.float64),
            "screen": np.array([r["screen"] for r in rows], dtype=np.int8),
            "reward": np.array([r["reward"] for r in rows], dtype=np.float32),
            "terminated": np.array([r["terminated"] for r in rows], dtype=bool),
            "flags": np.array([r["flags"] for r in rows], dtype=np.int16),
            "game_over_score": np.array([r["game_over_score"] for r in rows], dtype=np.float16),
            "z_fraction": np.array([r["z_fraction"] for r in rows], dtype=np.float16),
            "x_pressed": np.array([r["x_pressed"] for r in rows], dtype=bool),
            "esc_pressed": np.array([r["esc_pressed"] for r in rows], dtype=bool),
        }
        path = self.session_dir / f"part_{self._part:04d}.npz"
        self._part += 1

        def save() -> None:
            partial = path.with_name(path.stem + ".partial.npz")
            np.savez(partial, **arrays)
            os.replace(partial, path)

        if self._background:
            writer = threading.Thread(target=save)
            writer.start()
            self._writers.append(writer)
        else:
            save()

    def close(self) -> Dict[str, Any]:
        """残りを書いて、meta.json を書く"""
        if self._rows:
            self._write(self._rows)
            self._rows = []
        for writer in self._writers:
            writer.join()
        meta = dict(self.meta, frames=self._frames, parts=self._part, closed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        (self.session_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta


def iter_parts(session_dir: Path) -> Iterator[Dict[str, np.ndarray]]:
    """記録の part を順に読む (画像は大きいので、全体を一度にメモリに載せない使い方ができるよう、1 つずつ返す)"""
    for path in sorted(Path(session_dir).glob("part_*.npz")):
        if path.name.endswith(".partial.npz"):
            continue
        with np.load(path) as data:
            yield {name: data[name] for name in data.files}


def load_session(session_dir: Path, with_obs: bool = False) -> Dict[str, np.ndarray]:
    """記録全体を読む。画像は、with_obs=True のときだけ読む (数 GB になることがある)"""
    parts = [{k: v for k, v in part.items() if with_obs or k != "obs"} for part in iter_parts(session_dir)]
    if not parts:
        raise FileNotFoundError(f"記録が見つかりません: {session_dir}")
    return {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}


PLAY_SCREENS = frozenset({"playing", "chapter_finish", "spell_card_bonus", "stage_clear"})


def find_episodes(session: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    記録を、エピソード (プレイ中の連続したフレーム) に分ける

    始まり: プレイ画面 (オーバーレイを含む) になった最初のフレーム。
    終わり: 死亡 (ゲームオーバー画面の最初のフレーム) またはステージクリア (そのフレーム)、
    またはプレイ画面でなくなった (メニュー・ポーズなど) とき (reason = "left")。
    死亡の遷移の次の画面は、ゲームオーバー画面のフレーム。ステージクリアのあとは、次のステージの開始まで、新しいエピソードを始めない。
    """
    screens = [SCREEN_NAMES[i] for i in session["screen"]]
    flags = session["flags"]
    episodes: List[Dict[str, Any]] = []
    start: Optional[int] = None
    waiting_after_clear = False
    for k, name in enumerate(screens):
        playing = name in PLAY_SCREENS
        if start is None:
            if playing and not waiting_after_clear:
                start = k
            elif waiting_after_clear and name == "playing":
                waiting_after_clear = False
                start = k
            continue
        died = name == "game_over"
        cleared = bool(flags[k] & FLAG_STAGE_CLEAR)
        if died or cleared or not playing:
            end = k
            episodes.append({"start": start, "end": end, "reason": "death" if died else "clear" if cleared else "left",
                             "frames": end - start})
            start = None
            waiting_after_clear = cleared
    if start is not None:
        episodes.append({"start": start, "end": len(screens) - 1, "reason": "open", "frames": len(screens) - 1 - start})
    return episodes
