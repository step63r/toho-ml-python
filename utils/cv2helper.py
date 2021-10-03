import cv2
import numpy as np
import os


from typing import Any


def imread(filename: str, flags: int = cv2.IMREAD_COLOR, dtype=np.uint8) -> Any:
    """
    OpenCVで画像を読み込む（日本語対応）

    Parameters
    ----------
    filename : str
        ファイルパス
    flags : int, optional
        読み込みフラグ, by default cv2.IMREAD_COLOR
    dtype : [type], optional
        読み込む型, by default np.uint8

    Returns
    -------
    Any
        [description]
    """
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename: str, img: Any, params: Any=None) -> bool:
    """
    OpenCVで画像を書き出す（日本語対応）

    Parameters
    ----------
    filename : str
        ファイルパス
    img : Any
        書き出す画像
    params : Any, optional
        書き出し時パラメータ, by default None

    Returns
    -------
    bool
        成功したらTrue, 失敗したらFalse
    """
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
