"""
DQN で「東方紺珠伝」を学習する (実機のゲームにキー入力を送ります)

観測は、自機エリアの縮小グレー画像 (既定 0.25 倍 = 112x96) を 4 フレーム重ねたもの。
ゲームを起動しておき (どの画面からでもよい)、前面に置いたまま、他の操作をしないこと。ゲームは学習中も止まらない。

中断のしかた (どれでも、そこまでの学習を保存して終わる。--resume で続きから再開できる):
  - Ctrl+C
  - 出力ディレクトリに STOP という名前のファイルを作る (ゲームを前面に置いたままで止めたいとき):
        New-Item train_out\\default\\STOP

    python -m tools.train_dqn --total_steps 300000
    python -m tools.train_dqn --total_steps 300000 --resume train_out\\default\\interrupted.zip

ログ: 学習の経過 (SB3 の進捗の表も含む) は、ターミナルではなく <out_dir>/train.log に書く (ターミナルには、開始・終了・エラーだけを出す)。
再開しても同じファイルに追記する。見るには:
    Get-Content train_out\\default\\train.log -Wait -Tail 40

出力 (--out_dir):
    train.log   ログ (再開しても追記)
    checkpoints/dqn_<ステップ数>_steps.zip   定期保存 (モデルのみ)
    interrupted.zip と interrupted_replay_buffer/   中断・異常終了のとき (再開用。リプレイバッファは、埋まっている分を配列ごとの .npy で保存する)
    final.zip   目標のステップ数まで終わったとき
    monitor_<時刻>.monitor.csv   エピソードごとの 報酬 r (縮める前) / ステップ数 l / 経過秒 t
    tb/run_<時刻>/   tensorboard と progress.csv (SB3 の進捗の表と同じ値。tensorboard --logdir train_out)
"""
import argparse
import io
import json
import logging
import os
import shutil
import time

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import CSVOutputFormat, HumanOutputFormat, Logger, TensorBoardOutputFormat
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack

from environment.wrappers import ChannelLast, ScaledReward


logger = logging.getLogger(__name__)

STOP_FILE_NAME = "STOP"
LOG_FILE_NAME = "train.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# 再開時にリプレイバッファが無い場合に、バッファが貯まるまで学習を始めない steps 数
RESUME_WARMUP_STEPS = 5000
# リプレイバッファの保存対象 (ReplayBuffer の配列)
BUFFER_ARRAYS = ("observations", "actions", "rewards", "dones", "timeouts")


class LogStream(io.TextIOBase):
    """
    書き込まれた文字列を、flush のたびに 1 件のログとして出すストリーム

    SB3 の進捗の表 (HumanOutputFormat) は、複数行を書いてから flush する。
    1 行ずつログにすると、行ごとに時刻が付いて表が崩れるため、flush までをまとめて 1 件にする。
    """
    def __init__(self, target: logging.Logger, level: int = logging.INFO) -> None:
        super().__init__()
        self._target = target
        self._level = level
        self._chunks = []

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._chunks.append(text)
        return len(text)

    def flush(self) -> None:
        text = "".join(self._chunks).strip("\n")
        self._chunks.clear()
        if text:
            self._target.log(self._level, "\n%s", text)


def setup_logging(out_dir: Path, root: Optional[logging.Logger] = None) -> Path:
    """
    ログをファイル (追記) に出し、ターミナルにはエラーだけを出す

    他の既定のログ設定 (tools.common.open_window の basicConfig) より先に呼ぶこと。
    先に root にハンドラがあると、basicConfig は何もしない。
    """
    root = root or logging.getLogger()
    path = Path(out_dir) / LOG_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    for handler in (file_handler, console):
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.handlers[:] = [file_handler, console]
    root.setLevel(logging.INFO)
    return path


def make_sb3_logger(out_dir: Path) -> Logger:
    """SB3 の進捗の表をログファイルに、同じ値を progress.csv と tensorboard に出す"""
    folder = Path(out_dir) / "tb" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    return Logger(str(folder), [HumanOutputFormat(LogStream(logging.getLogger("sb3"))),
                                CSVOutputFormat(str(folder / "progress.csv")), TensorBoardOutputFormat(str(folder))])


class StopFileCallback(BaseCallback):
    """出力ディレクトリに STOP ファイルができたら、学習を止める (ゲームを前面に置いたまま、別のウィンドウから止められる)"""
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = Path(path)

    def _on_training_start(self) -> None:
        # 前回の停止指示が残っていると、始めた直後に止まってしまう
        self.path.unlink(missing_ok=True)

    def _on_step(self) -> bool:
        if self.path.exists():
            logger.warning("%s が見つかりました。学習を止めます。", self.path)
            return False
        return True


def build_vec_env(env_factory: Callable[[], gym.Env], out_dir: Path, n_stack: int = 4,
                  reward_scale: float = 0.01) -> VecEnv:
    """
    ゲームの環境を、学習用の観測 (チャンネル最後の画像を n_stack 枚重ねる) と報酬に包む

    Monitor は、報酬を縮める前の環境に付ける (記録される報酬を、ゲームの報酬のままにするため)。
    再開のたびに別のファイルにするのは、Monitor が既存のファイルを上書きするため。
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    monitor_path = Path(out_dir) / f"monitor_{time.strftime('%Y%m%d_%H%M%S')}"

    def make() -> gym.Env:
        env = Monitor(env_factory(), filename=str(monitor_path))
        return ScaledReward(ChannelLast(env), reward_scale)

    return VecFrameStack(DummyVecEnv([make]), n_stack=n_stack, channels_order="last")


def build_model(venv: VecEnv, out_dir: Path, params: Dict[str, Any]) -> DQN:
    # optimize_memory_usage: 次の観測を別に持たず、メモリを約半分にする (画像のバッファは大きいため)。
    # 時間切れの打ち切りを区別する仕組みと両立しないが、この環境は死亡かステージクリアでしか終わらないので問題ない
    model = DQN("CnnPolicy", venv, verbose=0, optimize_memory_usage=True,
                replay_buffer_kwargs={"handle_timeout_termination": False}, **params)
    model.set_logger(make_sb3_logger(out_dir))
    return model


def save_replay_buffer(buffer: ReplayBuffer, directory: Path) -> None:
    """
    リプレイバッファを、埋まっている分だけ、配列ごとの .npy にして保存する

    SB3 の save_replay_buffer (pickle) は、確保した全体 (今の設定で約 4.3GB) を 1 ファイルに書く。
    実機の学習の中断で、約 2.1GB まで書いたところで止まった (原因は未特定)。配列ごとに書けば、
    埋まっている分だけで済む。別名のディレクトリに書き切ってから置き換えるので、途中で止まっても、前回の保存は壊れない。
    """
    directory = Path(directory)
    partial = directory.with_name(directory.name + ".partial")
    old = directory.with_name(directory.name + ".old")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    size = buffer.size()
    for name in BUFFER_ARRAYS:
        np.save(partial / f"{name}.npy", getattr(buffer, name)[:size])
    (partial / "meta.json").write_text(json.dumps(
        {"pos": int(buffer.pos), "full": bool(buffer.full), "size": size, "buffer_size": int(buffer.buffer_size)}),
        encoding="utf-8")
    shutil.rmtree(old, ignore_errors=True)
    if directory.exists():
        os.replace(directory, old)
    os.replace(partial, directory)
    shutil.rmtree(old, ignore_errors=True)


def load_replay_buffer(buffer: ReplayBuffer, directory: Path) -> int:
    """save_replay_buffer で保存したバッファを、同じ大きさの buffer に読み込む。読み込んだ件数を返す"""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta["buffer_size"] != buffer.buffer_size:
        raise ValueError(f"バッファの大きさが違います (保存 {meta['buffer_size']} / 現在 {buffer.buffer_size})。")
    size = meta["size"]
    for name in BUFFER_ARRAYS:
        data = np.load(directory / f"{name}.npy", mmap_mode="r")      # 全体をメモリに二重に載せない
        target = getattr(buffer, name)
        if data.shape != (size,) + target.shape[1:]:
            raise ValueError(f"{name} の形が違います (保存 {data.shape} / 現在 {target.shape})。"
                             "--shrink_ratio や --n_stack が、保存したときと違う可能性があります。")
        target[:size] = data
    buffer.pos = int(meta["pos"])
    buffer.full = bool(meta["full"])
    return buffer.size()


def save_state(model: DQN, out_dir: Path, name: str, announce: Callable[[str], None] = print) -> Path:
    """
    モデルとリプレイバッファを保存する。モデルを先に保存し、リプレイバッファが保存できなくても、モデルは残る。
    どちらも別名に書いてから置き換える。リプレイバッファは数 GB あり、時間がかかるので、待つよう知らせる
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{name}.zip"
    model.save(out_dir / f"{name}.partial.zip")
    os.replace(out_dir / f"{name}.partial.zip", model_path)
    logger.info("モデルを保存しました: %s (ステップ数 %d)", model_path, model.num_timesteps)

    buffer = model.replay_buffer
    gigabytes = sum(getattr(buffer, n)[:buffer.size()].nbytes for n in BUFFER_ARRAYS) / 1e9
    announce(f"モデルを保存しました。リプレイバッファ (約 {gigabytes:.1f} GB) を保存しています。"
             "完了するまで、もう一度 Ctrl+C を押さないでください。")
    try:
        save_replay_buffer(buffer, out_dir / f"{name}_replay_buffer")
    except Exception:
        logger.exception("リプレイバッファを保存できませんでした。モデルだけを保存しました。")
        announce("リプレイバッファを保存できませんでした (モデルは保存済み)。再開すると、バッファが貯まるまで学習を始めません。")
        return model_path
    logger.info("リプレイバッファを保存しました: %s", out_dir / f"{name}_replay_buffer")
    return model_path


def load_for_resume(path: Path, venv: VecEnv, out_dir: Path, device: str = "auto",
                    train_freq: Optional[int] = None, gradient_steps: Optional[int] = None) -> DQN:
    """
    保存したモデルを読み込む。リプレイバッファが隣にあれば読み込み、無ければ貯まるまで学習を始めない

    train_freq と gradient_steps を渡すと、保存したモデルの値を上書きする (再開では、保存した値が使われるため)。
    """
    path = Path(path)
    model = DQN.load(path, env=venv, device=device)
    model.verbose = 0                                # 保存したときの値 (1) だと、ターミナルに出力してしまう
    model.set_logger(make_sb3_logger(out_dir))
    if train_freq is not None:
        model.train_freq = TrainFreq(int(train_freq), TrainFrequencyUnit.STEP)
    if gradient_steps is not None:
        model.gradient_steps = int(gradient_steps)
    logger.info("学習の設定: %d ステップごとに、勾配更新を %d 回", model.train_freq.frequency, model.gradient_steps)

    buffer_dir = path.with_name(path.stem + "_replay_buffer")
    legacy_path = path.with_name(path.stem + "_replay_buffer.pkl")
    if (buffer_dir / "meta.json").exists():
        count = load_replay_buffer(model.replay_buffer, buffer_dir)
        logger.info("リプレイバッファを読み込みました: %s (%d 件)", buffer_dir, count)
    elif legacy_path.exists():
        model.load_replay_buffer(legacy_path)
        logger.info("リプレイバッファを読み込みました: %s (%d 件)", legacy_path, model.replay_buffer.size())
    else:
        model.learning_starts = model.num_timesteps + RESUME_WARMUP_STEPS
        logger.warning("リプレイバッファが無いので、%d ステップ貯まるまで学習を始めません (ステップ数 %d から)。",
                       RESUME_WARMUP_STEPS, model.learning_starts)
    return model


def train(model: DQN, out_dir: Path, total_steps: int, checkpoint_every: int) -> str:
    """
    total_steps (累計のステップ数) まで学習する

    Returns
    -------
    str
        "final" (目標まで終わった) / "interrupted" (Ctrl+C か STOP ファイル)。
        異常終了のときは、保存してから例外をそのまま送出する
    """
    out_dir = Path(out_dir)
    callbacks = CallbackList([
        CheckpointCallback(save_freq=checkpoint_every, save_path=str(out_dir / "checkpoints"), name_prefix="dqn"),
        StopFileCallback(out_dir / STOP_FILE_NAME),
    ])
    remaining = total_steps - model.num_timesteps
    if remaining <= 0:
        logger.info("すでに目標のステップ数 (%d) に達しています。", total_steps)
        return "final"
    try:
        # reset_num_timesteps=False では、SB3 が現在のステップ数を加えるので、残りを渡す
        # (探索率の計画が、再開しても元の通りに続く)
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)
    except KeyboardInterrupt:
        logger.warning("中断しました。")
        save_state(model, out_dir, "interrupted")
        return "interrupted"
    except BaseException:
        logger.exception("異常終了しました。ここまでの学習を保存します。")
        save_state(model, out_dir, "interrupted")
        raise
    if model.num_timesteps >= total_steps:
        save_state(model, out_dir, "final")
        return "final"
    save_state(model, out_dir, "interrupted")
    return "interrupted"


def model_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "learning_rate": args.learning_rate, "buffer_size": args.buffer_size, "learning_starts": args.learning_starts,
        "batch_size": args.batch_size, "gamma": args.gamma,
        "train_freq": 4 if args.train_freq is None else args.train_freq,
        "gradient_steps": 1 if args.gradient_steps is None else args.gradient_steps,
        "target_update_interval": args.target_update_interval, "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps, "device": args.device, "seed": args.seed,
    }


def check_start_paths(out_dir: Path, resume: Optional[Path]) -> Optional[str]:
    """
    学習を始める前 (ゲームにキーを送る前) の、保存先と再開元の検査。問題があれば、そのメッセージを返す

    再開元が無いのに始めると、メニューを進めたあとで失敗する。再開せずに保存のある出力先で始めると、
    次の中断で前回の保存 (interrupted.zip) を上書きしてしまう。
    """
    out_dir = Path(out_dir)
    if resume is not None:
        if not Path(resume).is_file():
            saved = sorted(p.name for p in out_dir.glob("*.zip")) + sorted(
                f"checkpoints\\{p.name}" for p in (out_dir / "checkpoints").glob("*.zip"))
            return (f"--resume のファイルが見つかりません: {resume}\n"
                    "interrupted.zip は、学習を中断または異常終了したときに作られます。最初の学習は --resume なしで実行してください。\n"
                    + (f"{out_dir} にあるモデル: {', '.join(saved)}" if saved else f"{out_dir} にはまだモデルがありません。"))
        return None
    existing = [name for name in ("interrupted.zip", "final.zip") if (out_dir / name).exists()]
    if existing:
        return (f"{out_dir} には、すでに保存があります ({', '.join(existing)})。\n"
                f"続きから学習するなら --resume {out_dir / existing[0]} を付けてください。"
                "最初からやり直すなら、--out_dir を変えるか、保存を別の場所に移してください (上書きを避けるため、始めません)。")
    return None


def main() -> None:
    from environment.shanghai_alice_15th import Kanjuden
    from tools.common import add_window_args, open_window

    parser = argparse.ArgumentParser(description="DQN で学習する (ゲームにキー入力を送ります。Ctrl+C か STOP ファイルで中断)")
    add_window_args(parser)
    parser.set_defaults(shrink_ratio=0.25)           # 450x384 → 112x96。弾を見分けられる最小限の大きさとして選んだ (未検証)
    parser.add_argument("--out_dir", type=Path, default=Path("train_out") / "default")
    parser.add_argument("--total_steps", type=int, default=300_000,
                        help="累計のステップ数。1 ステップは約 0.13 秒 (約 7.5 steps/s) で、30 万ステップは約 11 時間")
    parser.add_argument("--resume", type=Path, default=None, help="保存したモデル (.zip) から続きを学習する")
    parser.add_argument("--checkpoint_every", type=int, default=10_000)
    parser.add_argument("--n_stack", type=int, default=4)
    parser.add_argument("--reward_scale", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=100_000, help="1 件あたり約 43KB (112x96x4)。10 万件で約 4.3GB")
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train_freq", type=int, default=None,
                        help="何ステップごとに学習するか (既定 4)。再開で指定すると、保存した値を上書きする")
    parser.add_argument("--gradient_steps", type=int, default=None,
                        help="学習のたびの勾配更新の回数 (既定 1)。再開で指定すると、保存した値を上書きする。"
                             "ゲームの 1 ステップは約 0.13 秒で、GPU の 1 回の更新は数 ms なので、増やしても速度はほとんど落ちない")
    parser.add_argument("--target_update_interval", type=int, default=2000)
    parser.add_argument("--exploration_fraction", type=float, default=0.3,
                        help="探索率を下げていく範囲 (全ステップに対する割合)")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    problem = check_start_paths(args.out_dir, args.resume)
    if problem:
        parser.error(problem)                        # ゲームにキーを送る前に終わる
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(args.out_dir)           # open_window の basicConfig より先に
    open_window(args)                                # ウィンドウの存在確認
    (args.out_dir / f"config_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    def env_factory() -> gym.Env:
        return Kanjuden(app_class_name=args.app_class_name, app_window_name=args.app_window_name,
                        shrink_ratio=args.shrink_ratio, clip_top=args.clip_top, clip_bottom=args.clip_bottom,
                        clip_left=args.clip_left, clip_right=args.clip_right,
                        diagnostics_dir=str(Path("diagnostics_out")))

    venv = build_vec_env(env_factory, args.out_dir, args.n_stack, args.reward_scale)
    if args.resume:
        model = load_for_resume(args.resume, venv, args.out_dir, args.device, args.train_freq, args.gradient_steps)
    else:
        model = build_model(venv, args.out_dir, model_params(args))
    print(f"=== 学習 (ゲームにキー入力を送ります。Ctrl+C または {args.out_dir / STOP_FILE_NAME} の作成で中断)")
    print(f"    現在 {model.num_timesteps} / 目標 {args.total_steps} ステップ。device={model.device}")
    print(f"    ログ: {log_path}")
    try:
        result = train(model, args.out_dir, args.total_steps, args.checkpoint_every)
    finally:
        venv.close()                                 # 押しっぱなしのキーを離す
    print(f"終了: {result} (ステップ数 {model.num_timesteps})")


if __name__ == "__main__":
    main()
