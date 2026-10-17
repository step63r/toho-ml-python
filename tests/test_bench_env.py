import pytest

from environment.screens import Screen
from tests.test_env import FakeWindow, keys, make_env, queue_game_over, step_with, overlay_frame, STAGE_CLEAR_RECT  # noqa: F401
from tools import bench_env


def test_ゲームオーバーまでの生存時間とステップ数を記録する(keys):
    env = make_env()
    counter = {"steps": 0}

    def policy(observation):
        counter["steps"] += 1
        if counter["steps"] in (4, 9):
            queue_game_over()          # 次の入力の直後にゲームオーバー画面になる
        return 0

    report = bench_env.run_bench(env, episodes=2, max_episode_seconds=30, max_seconds=60, policy=policy)

    assert [e["ended"] for e in report["episodes"]] == ["game_over", "game_over"]
    assert [e["steps"] for e in report["episodes"]] == [4, 5]
    assert report["ended"] == {"game_over": 2, "stage_clear": 0, "time": 0}
    assert report["event_counts"] == {"game_over": 2}
    assert report["survival_seconds"]["max"] >= report["survival_seconds"]["min"] > 0
    assert report["steps_per_second"] > 0


def test_内部処理の所要時間も測る(keys):
    env = make_env()

    report = bench_env.run_bench(env, episodes=1, max_episode_seconds=0.3, max_seconds=5, policy=lambda o: 0)

    assert set(report["timing"]) == {"capture", "detect"}
    assert report["timing"]["detect"]["count"] > 0
    assert report["ended"]["time"] == 1


def test_打ち切りと全体の制限時間(keys):
    env = make_env()

    report = bench_env.run_bench(env, episodes=100, max_episode_seconds=0.1, max_seconds=0.35, policy=lambda o: 0)

    # 全体の制限時間で止まり、100回までは回らない。打ち切りのエピソードは ended="time"
    assert 1 <= len(report["episodes"]) < 100
    assert all(e["ended"] == "time" for e in report["episodes"])


def test_ステージクリアは別に数える(keys):
    env = make_env()
    FakeWindow.instance.gray = overlay_frame("stage_clear", STAGE_CLEAR_RECT)

    report = bench_env.run_bench(env, episodes=1, max_episode_seconds=5, max_seconds=10, policy=lambda o: 0)

    assert report["ended"]["stage_clear"] == 1
    assert "stage_clear" in report["event_counts"]


def test_結果が空でも要約できる():
    report = bench_env.summarize([], {}, 1.0)

    assert report["survival_seconds"] == {} and report["steps_per_second"] == 0.0
