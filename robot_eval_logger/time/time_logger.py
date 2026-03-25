import time
from threading import Event, Thread, current_thread
from typing import Optional

import wandb


class TimeLogger:
    """Periodically logs step-throughput statistics to wandb in a background thread."""

    def __init__(self, wandb_logger, interval_minutes: float):
        self._wandb_logger = wandb_logger
        self._interval_minutes = interval_minutes

        self.total_steps = 0
        self.steps_since_last_log = 0
        self.time_elapsed = 0.0
        self.last_log_time: Optional[float] = None

        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    def start(self):
        """Start the periodic logging background thread."""
        if self._thread is None:
            self._stop_event.clear()
            self._thread = Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        """Safely stop the background thread."""
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            if self._thread != current_thread():
                self._thread.join(timeout=1.0)
            self._thread = None

    def record_step(self):
        """Increment step counters (call once per eval step)."""
        self.total_steps += 1
        self.steps_since_last_log += 1

    def _run(self):
        """Background loop that calls :meth:`_log_stats` at the configured interval."""
        while not self._stop_event.is_set():
            try:
                self._log_stats()
            except Exception as e:
                print(f"Error in periodic logging: {e}")
                break
            self._stop_event.wait(timeout=self._interval_minutes * 60)
        self._thread = None

    def _log_stats(self):
        """Compute and push time-related metrics to wandb."""
        current_time = time.time()

        if self.last_log_time is None:
            self.last_log_time = current_time
            return

        minutes_elapsed = (current_time - self.last_log_time) / 60.0
        self.time_elapsed += minutes_elapsed

        steps_per_minute = (
            self.steps_since_last_log / minutes_elapsed if minutes_elapsed > 0 else 0
        )

        wandb.define_metric("step_stats/*", step_metric="total_time_elapsed")
        self._wandb_logger.log(
            {
                "step_stats/total_eval_steps": self.total_steps,
                "step_stats/eval_steps_per_minute": steps_per_minute,
                "total_time_elapsed": self.time_elapsed,
            }
        )

        self.last_log_time = current_time
        self.steps_since_last_log = 0
