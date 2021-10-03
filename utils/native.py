import ctypes


PUL = ctypes.POINTER(ctypes.c_ulong)


# Scan code maps
SCAN_ESC    = 0x01
SCAN_LSHIFT = 0x2A
SCAN_Z      = 0x2C
SCAN_UP     = 0x48      # 0xC8
SCAN_LEFT   = 0x4B      # 0xCB
SCAN_RIGHT  = 0x4D      # 0xCD
SCAN_DOWN   = 0x50      # 0xD0


# サーバレスポンスに対応する操作
COMMAND_DICT = {
    -1: [SCAN_ESC],
    0: [SCAN_Z],
    1: [SCAN_Z, SCAN_LEFT],
    2: [SCAN_Z, SCAN_UP],
    3: [SCAN_Z, SCAN_RIGHT],
    4: [SCAN_Z, SCAN_DOWN],
    5: [SCAN_Z, SCAN_LEFT, SCAN_UP],
    6: [SCAN_Z, SCAN_UP, SCAN_RIGHT],
    7: [SCAN_Z, SCAN_RIGHT, SCAN_DOWN],
    8: [SCAN_Z, SCAN_DOWN, SCAN_LEFT],
    9: [SCAN_LSHIFT, SCAN_Z],
    10: [SCAN_LSHIFT, SCAN_Z, SCAN_LEFT],
    11: [SCAN_LSHIFT, SCAN_Z, SCAN_UP],
    12: [SCAN_LSHIFT, SCAN_Z, SCAN_RIGHT],
    13: [SCAN_LSHIFT, SCAN_Z, SCAN_DOWN],
    14: [SCAN_LSHIFT, SCAN_Z, SCAN_LEFT, SCAN_UP],
    15: [SCAN_LSHIFT, SCAN_Z, SCAN_UP, SCAN_RIGHT],
    16: [SCAN_LSHIFT, SCAN_Z, SCAN_RIGHT, SCAN_DOWN],
    17: [SCAN_LSHIFT, SCAN_Z, SCAN_DOWN, SCAN_LEFT],
}


# C struct redefinitions
class KeyBdInput(ctypes.Structure):
    """
    KEYBDINPUT構造体 (winuser.h)
        WORD        wVk;
        WORD        wScan;
        DWORD       dwFlags;
        DWORD       time;
        ULONG_PTR   dwExtraInfo;

    Parameters
    ----------
    ctypes : [type]
        [description]
    """
    _fields_ = [('wVk', ctypes.c_ushort),
                ('wScan', ctypes.c_ushort),
                ('dwFlags', ctypes.c_ulong),
                ('time', ctypes.c_ulong),
                ('dwExtraInfo', PUL)]


class HardwareInput(ctypes.Structure):
    """
    HARDWAREINPUT構造体 (winuser.h)
        DWORD       uMsg;
        WORD        wParamL;
        DWORD       wParamH;

    Parameters
    ----------
    ctypes : [type]
        [description]
    """
    _fields_ = [('uMsg', ctypes.c_ulong),
                ('wParamL', ctypes.c_short),
                ('wParamH', ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    """
    MOUSEINPUT構造体 (winuser.h)
        LONG        dx;
        LONG        dy;
        DWORD       mouseData;
        DWORD       dwFlags;
        DWORD       time;
        ULONG_PTR   dwExtraInfo;

    Parameters
    ----------
    ctypes : [type]
        [description]
    """
    _fields_ = [('dx', ctypes.c_long),
                ('dy', ctypes.c_long),
                ('mouseData', ctypes.c_ulong),
                ('dwFlags', ctypes.c_ulong),
                ('time', ctypes.c_ulong),
                ('dwExtraInfo', PUL)]


class Input_I(ctypes.Union):
    """
    INPUT構造体の共用体部分
        MOUSEINPUT      mi;
        KEYBDINPUT      ki;
        HARDWAREINPUT   hi;

    Parameters
    ----------
    ctypes : [type]
        [description]
    """
    _fields_ = [('ki', KeyBdInput),
                ('mi', MouseInput),
                ('hi', HardwareInput)]


class Input(ctypes.Structure):
    """
    INPUT構造体 (winuser.h)
        DWORD type;
        union {
            MOUSEINPUT      mi;
            KEYBDINPUT      ki;
            HARDWAREINPUT   hi;
        };

    Parameters
    ----------
    ctypes : [type]
        [description]
    """
    _fields_ = [('type', ctypes.c_ulong),
                ('ii', Input_I)]


def PressKey(hex_key_code: int) -> None:
    """
    キーを押下する

    Parameters
    ----------
    hex_key_code : int
        キーコード
    """
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hex_key_code, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hex_key_code: int) -> None:
    """
    キーを離す

    Parameters
    ----------
    hex_key_code : int
        キーコード
    """
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


class Rect(ctypes.Structure):
    """
    RECT構造体 (windef.h)
        LONG        left;
        LONG        top;
        LONG        right;
        LONG        bottom;
    
    Parameters
    ----------
    ctypes : [type]
        [description]
    """
    _fields_ = [('left', ctypes.c_long),
                ('top', ctypes.c_long),
                ('right', ctypes.c_long),
                ('bottom', ctypes.c_long)]


class WindowInfo(ctypes.Structure):
    """
    WINDOWINFO構造体 (winuser.h)
        DWORD       cbSize;
        RECT        rcWindow;
        RECT        rcClient;
        DWORD       dwStyle;
        DWORD       dwExStyle;
        DWORD       dwWindowStatus;
        UINT        cxWindowBorders;
        UINT        cyWindowBorders;
        ATOM        atomWindowType;
        WORD        wCreatorVersion;

    Parameters
    ----------
    ctypes : [type]
        [description]
    """
    _fields_ = [('cbSize', ctypes.c_ulong),
                ('rcWindow', Rect),
                ('rcClient', Rect),
                ('dwStyle', ctypes.c_ulong),
                ('dwExStyle', ctypes.c_ulong),
                ('dwWindowStatus', ctypes.c_ulong),
                ('cxWindowBorders', ctypes.c_uint),
                ('cyWindowBorders', ctypes.c_uint),
                ('atomWindowType', ctypes.c_ushort),
                ('wCreatorVersion', ctypes.c_ushort)]


def GetWindowRect(class_name: str, window_name: str) -> tuple[int, int, int, int]:
    """
    指定したクラス名のウィンドウ座標を取得する

    Parameters
    ----------
    className : str
        クラス名
    window_name : str
        ウィンドウキャプション

    Returns
    -------
    tuple[int, int, int, int]
        ウィンドウ座標 (左, 上, 右, 下)

    Raises
    ------
    WindowsError
        FindWindow() 失敗
    WindowsError
        GetWindowInfo() 失敗
    """
    hWnd: ctypes.c_void_p = ctypes.windll.user32.FindWindowW(ctypes.c_wchar_p(class_name), ctypes.c_wchar_p(window_name))
    if not hWnd:
        raise WindowsError("FindWindow() 失敗")
    window_info = WindowInfo()
    bRet: ctypes.c_bool = ctypes.windll.user32.GetWindowInfo(hWnd, ctypes.byref(window_info))
    if not bRet:
        raise WindowsError("GetWindowInfo() 失敗")
    return window_info.rcWindow.left, window_info.rcWindow.top, window_info.rcWindow.right, window_info.rcWindow.bottom
