import json
import logging
import time

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from utils import cv2helper


logger = logging.getLogger(__name__)


class FrameRecorder:
    """
    直近のフレームと、その時点の判定・動き量を溜めておき、問題が起きたときに書き出す

    迷子になった原因 (そのとき何が画面に映り、どう判定され、どれだけ動いていたか) は、
    ログの画面名だけでは分からない。常時ディスクに書くと重いので、メモリに溜めて、必要なときだけ書く。
    """
    def __init__(self, out_dir: Path, capacity: int = 30) -> None:
        self.out_dir = Path(out_dir)
        self._frames: Deque[Tuple[float, np.ndarray, Dict[str, Any]]] = deque(maxlen=capacity)
        self._start = time.monotonic()

    def record(self, gray: np.ndarray, **info: Any) -> None:
        # 取得のたびに新しい配列が作られる前提で、コピーせず参照だけ持つ
        self._frames.append((time.monotonic() - self._start, gray, info))

    def dump(self, reason: str) -> Optional[Path]:
        """溜めたフレームを out_dir/<時刻>_<reason>/ に書き出す。溜まっていなければ何もしない"""
        if not self._frames:
            return None
        target = self.out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{reason}"
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, (elapsed, gray, info) in enumerate(self._frames):
            name = f"{index:03d}.png"
            cv2helper.imwrite(str(target / name), gray)
            rows.append({"file": name, "t": round(elapsed, 3), **info})
        (target / "frames.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.warning("診断用のフレームを保存しました (%s): %s", reason, target)
        return target
