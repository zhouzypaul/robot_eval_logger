"""Tests for :class:`robot_eval_logger.time.TimeLogger` and EvalLogger wiring.

How to run (from the repository root)::

    python -m pytest tests/test_time_logger.py -v -o addopts=
"""
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robot_eval_logger.eval_logger import EvalLogger
from robot_eval_logger.time import TimeLogger


def _minimal_log_step_kwargs():
    """Valid arguments for :meth:`EvalLogger.log_step` (one 7-DOF joint arm)."""
    obs = {"image_primary": np.zeros((2, 2, 3), dtype=np.uint8)}
    z = np.zeros(7, dtype=np.float32)
    return {
        "obs": obs,
        "action": z,
        "joint_position": z,
        "joint_velocity": z,
        "end_effector_pose": z,
        "gripper": z[:1],
    }


class TestTimeLoggerRecordStep:
    def test_record_step_increments_counters(self):
        wandb_mock = MagicMock()
        tl = TimeLogger(wandb_mock, interval_minutes=1.0)
        for _ in range(7):
            tl.record_step()
        assert tl.total_steps == 7
        assert tl.steps_since_last_log == 7


class TestTimeLoggerLogStats:
    @patch("robot_eval_logger.time.time_logger.wandb.define_metric")
    def test_first_call_initializes_clock_only(self, mock_define):
        wandb_mock = MagicMock()
        tl = TimeLogger(wandb_mock, interval_minutes=1.0)
        with patch(
            "robot_eval_logger.time.time_logger.time.time", return_value=1_000.0
        ):
            tl._log_stats()
        assert tl.last_log_time == 1_000.0
        wandb_mock.log.assert_not_called()
        mock_define.assert_not_called()

    @patch("robot_eval_logger.time.time_logger.wandb.define_metric")
    def test_second_call_logs_steps_per_minute_and_elapsed_time(self, mock_define):
        wandb_mock = MagicMock()
        tl = TimeLogger(wandb_mock, interval_minutes=1.0)
        with patch(
            "robot_eval_logger.time.time_logger.time.time", return_value=1_000.0
        ):
            tl._log_stats()
        for _ in range(40):
            tl.record_step()
        with patch(
            "robot_eval_logger.time.time_logger.time.time",
            return_value=1_000.0 + 60.0,
        ):
            tl._log_stats()
        wandb_mock.log.assert_called_once()
        payload = wandb_mock.log.call_args[0][0]
        assert payload["step_stats/total_eval_steps"] == 40
        assert payload["step_stats/eval_steps_per_minute"] == pytest.approx(40.0)
        assert payload["total_time_elapsed"] == pytest.approx(1.0)
        mock_define.assert_called_once_with(
            "step_stats/*", step_metric="total_time_elapsed"
        )
        assert tl.steps_since_last_log == 0
        assert tl.time_elapsed == pytest.approx(1.0)

    @patch("robot_eval_logger.time.time_logger.wandb.define_metric")
    def test_zero_minutes_elapsed_yields_zero_steps_per_minute(self, mock_define):
        wandb_mock = MagicMock()
        tl = TimeLogger(wandb_mock, interval_minutes=1.0)
        with patch("robot_eval_logger.time.time_logger.time.time", return_value=500.0):
            tl._log_stats()
        tl.record_step()
        tl.record_step()
        with patch("robot_eval_logger.time.time_logger.time.time", return_value=500.0):
            tl._log_stats()
        payload = wandb_mock.log.call_args[0][0]
        assert payload["step_stats/eval_steps_per_minute"] == 0.0
        assert payload["step_stats/total_eval_steps"] == 2


class TestTimeLoggerThreading:
    def test_stop_without_start_is_safe(self):
        tl = TimeLogger(MagicMock(), interval_minutes=1.0)
        tl.stop()

    def test_start_is_idempotent(self):
        wandb_mock = MagicMock()
        tl = TimeLogger(wandb_mock, interval_minutes=60.0)
        tl.start()
        first = tl._thread
        assert first is not None
        tl.start()
        assert tl._thread is first
        tl.stop()
        assert tl._thread is None

    def test_stop_joins_background_thread(self):
        wandb_mock = MagicMock()
        tl = TimeLogger(wandb_mock, interval_minutes=60.0)
        tl.start()
        time.sleep(0.1)
        tl.stop()
        assert tl._thread is None

    @patch("robot_eval_logger.time.time_logger.wandb.define_metric")
    def test_wandb_log_error_exits_loop(self, mock_define):
        wandb_mock = MagicMock()
        wandb_mock.log.side_effect = RuntimeError("wandb unavailable")

        tl = TimeLogger(wandb_mock, interval_minutes=0.0)
        with patch("robot_eval_logger.time.time_logger.time.time", return_value=900.0):
            tl.start()
        time.sleep(0.2)
        tl.stop()
        wandb_mock.log.assert_called()


class TestEvalLoggerTimeLogging:
    def test_creates_time_logger_when_wandb_and_interval_set(self):
        wandb_mock = MagicMock()
        logger = EvalLogger(
            wandb_logger=wandb_mock, log_step_stats_interval_minutes=5.0
        )
        assert logger._time_logger is not None
        logger.stop_time_logging()

    def test_no_time_logger_without_interval(self):
        logger = EvalLogger(
            wandb_logger=MagicMock(), log_step_stats_interval_minutes=None
        )
        assert logger._time_logger is None

    def test_no_time_logger_without_wandb(self):
        logger = EvalLogger(wandb_logger=None, log_step_stats_interval_minutes=1.0)
        assert logger._time_logger is None

    def test_log_step_increments_time_logger_steps(self):
        wandb_mock = MagicMock()
        logger = EvalLogger(
            wandb_logger=wandb_mock, log_step_stats_interval_minutes=999.0
        )
        kw = _minimal_log_step_kwargs()
        logger.log_step(**kw)
        logger.log_step(**kw)
        assert logger._time_logger.total_steps == 2
        assert logger._time_logger.steps_since_last_log == 2
        logger.stop_time_logging()
