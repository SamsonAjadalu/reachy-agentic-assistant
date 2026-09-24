"""Tests for app-level runtime behavior."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import reachy_mini_conversation_app.main as main_mod


def test_settings_ui_url_is_loopback_only() -> None:
    """The dashboard remains a local service."""
    assert main_mod.ReachyMiniConversationApp.custom_app_url == "http://127.0.0.1:7860/"


def test_direct_run_enables_motors_before_startup_movement(monkeypatch) -> None:
    """The direct CLI must enable torque before issuing startup movement commands."""

    class StopAfterStartup(Exception):
        pass

    robot = MagicMock()
    monkeypatch.setattr(main_mod, "ReachyMini", MagicMock(return_value=robot))

    def stop_after_startup(startup_robot, _logger) -> None:
        assert startup_robot is robot
        robot.enable_motors.assert_called_once_with()
        raise StopAfterStartup()

    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", stop_after_startup)
    args = SimpleNamespace(command=None, debug=False, robot_name=None, no_camera=False, ui=False)

    try:
        main_mod.run(args)
    except StopAfterStartup:
        pass
    else:
        raise AssertionError("run() did not reach the startup movement")


def test_inactivity_timeout_thread_goes_to_sleep() -> None:
    """The watchdog should use the shared sleep shutdown path once activity is too old."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())
    go_to_sleep = MagicMock(return_value={"status": "sleeping"})

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
        go_to_sleep=go_to_sleep,
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    go_to_sleep.assert_called_once_with()
    stream_manager.close.assert_not_called()


def test_inactivity_timeout_thread_closes_stream_manager_without_sleep_callback() -> None:
    """The watchdog should still close the stream when no sleep callback is available."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    stream_manager.close.assert_called_once_with()
