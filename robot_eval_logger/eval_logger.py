import time
from threading import Event, Thread, current_thread
from typing import Optional

import numpy as np
import wandb

from robot_eval_logger.storage import BaseSaver, LocalStorage
from robot_eval_logger.typing import *


class EvalLogger:
    """
    Class to log robot evaluation metrics. This is the main interface to interact with the logger.
    This class offers the following main APIs:
        - log_step: log all metrics for a single step, users can call this at the end of every step
        - log_episode: log all metrics for an entire episode, users can all this at the end of every episode
        - save_metadata: save metadata about the evaluation

    The functionality the EvalLogger offers are:
        - log: log all related metrics
        - storage: store the data locally or on the cloud
        - visualize: visualize the logged data (e.g. with a wandb report)
    """

    def __init__(
        self,
        wandb_logger=None,
        frames_visualizer=None,
        data_saver: Optional[BaseSaver] = None,
        log_step_stats_interval_minutes: Optional[float] = None,
    ):
        self.wandb_logger = wandb_logger
        self.frames_visualizer = frames_visualizer
        self.data_saver = data_saver
        self.log_step_stats_interval_minutes = log_step_stats_interval_minutes

        self.past_success_rates = {}  # prefix --> list of success data points
        self.metadata_saved = False

        # step tracking
        self.total_steps = 0
        self.steps_since_last_log = 0
        self.time_elapsed = 0.0
        self.last_log_time = None

        # episode tracking
        self.current_episode = 0

        # set up periodic logging
        # TODO(zhouzypaul): make this into a separate class the logger can take in instead
        if self.wandb_logger is not None:
            self._time_logging = Event()
            self._time_logging_thread = None
            if log_step_stats_interval_minutes is not None:
                self._start_time_logging_thread()

    def _start_time_logging_thread(self):
        """Start the time logging in a background thread"""
        if self._time_logging_thread is None:
            self._time_logging.clear()
            self._time_logging_thread = Thread(
                target=self._periodic_logging, daemon=True
            )
            self._time_logging_thread.start()

    def stop_time_logging(self):
        """Safely stop the time logging thread"""
        if (
            self._time_logging_thread is not None
            and self._time_logging_thread.is_alive()
        ):
            self._time_logging.set()
            if self._time_logging_thread != current_thread():
                self._time_logging_thread.join(timeout=1.0)
            self._time_logging_thread = None

    def __del__(self):
        """Clean up the time logging thread when the logger is destroyed and log any remaining frames"""
        try:
            # First, log any remaining frames that haven't been logged yet
            self._visualize_remaining_frames()

            # Then stop the time logging thread
            self.stop_time_logging()
        except Exception as e:
            print(f"Error in EvalLogger.__del__: {e}")
            pass  # ignore any errors during cleanup

    def _visualize_remaining_frames(self):
        """Log any remaining frames that haven't been logged due to not reaching the periodic threshold"""
        if self.frames_visualizer is None:
            return

        # Log the remaining frames
        frames_viz = self.frames_visualizer.log_remaining_frames(
            final_step=self.current_episode, success_rates=self.past_success_rates
        )

        # Push to wandb if there's anything to log
        if frames_viz and self.wandb_logger is not None:
            self.wandb_logger.log({**frames_viz, "num_episode": self.current_episode})

    def _periodic_logging(self):
        """Background thread that periodically logs time-related stats"""
        while not self._time_logging.is_set():
            try:
                self.log_time_related_stats()
            except Exception as e:
                print(f"Error in periodic logging: {e}")
                # if we hit a logging error, stop the thread
                break
            # sleep for the interval, but also check if we should stop
            self._time_logging.wait(timeout=self.log_step_stats_interval_minutes * 60)

        # make sure thread is marked as stopped
        self._time_logging_thread = None

    def log_time_related_stats(self):
        """Log time-related statistics to wandb"""
        if self.wandb_logger is None:
            return

        try:
            current_time = time.time()

            # initialize last_log_time if not set
            if self.last_log_time is None:
                self.last_log_time = current_time
                return

            # calculate elapsed time and stats
            minutes_elapsed = (current_time - self.last_log_time) / 60.0
            self.time_elapsed += minutes_elapsed

            # calculate steps per minute in the last interval
            steps_per_minute = (
                self.steps_since_last_log / minutes_elapsed
                if minutes_elapsed > 0
                else 0
            )

            # change the x-axis on wandb
            wandb.define_metric("step_stats/*", step_metric="total_time_elapsed")
            self.wandb_logger.log(
                {
                    "step_stats/total_eval_steps": self.total_steps,
                    "step_stats/eval_steps_per_minute": steps_per_minute,
                    "total_time_elapsed": self.time_elapsed,
                }
            )

            # reset tracking variables
            self.last_log_time = current_time
            self.steps_since_last_log = 0
        except Exception as e:
            print(f"Error logging time-related stats: {e}")
            raise  # re-raise to stop the thread if needed

    def log_episode(
        self,
        i_episode,
        logging_prefix,
        episode_success,
        frames_to_log=None,
        **kwargs,
    ):
        """log at the end of an eval episode"""
        self.current_episode = i_episode
        # log success rate
        success_stats = self.log_success_rates(
            step=i_episode,
            logging_prefix=logging_prefix,
            episode_success=episode_success,
        )  # this before log_frames, so frames-with-success is visualized correctly
        # log frames visualization
        if self.frames_visualizer is not None:
            frames_viz = self.frames_visualizer.log_frames(
                step=i_episode,
                logging_prefix=logging_prefix,
                frames=frames_to_log,
                success_rates=self.past_success_rates[logging_prefix],
            )
        else:
            frames_viz = {}
        # log all others
        others = {f"{logging_prefix}/{k}": v for k, v in kwargs.items()}
        to_log = {**frames_viz, **success_stats, **others}

        # push to wandb
        if self.wandb_logger is not None:
            # change the x-axis on wandb
            assert (
                logging_prefix is not None
            ), "Doesn't support logging without a prefix currently"
            for k in to_log.keys():
                wandb.define_metric(f"{k}/*", step_metric="num_episode")
            self.wandb_logger.log({**to_log, "num_episode": i_episode})

        # save the data for this episode
        if self.data_saver:
            self.data_saver.save_episode(
                i_episode=i_episode,
                language_command=logging_prefix,
                obs={"image_primary": frames_to_log},
                success=episode_success,
                **kwargs,
            )

        return to_log

    def log_step(
        self,
    ):
        """log at the end of an action step during an eval episode"""
        # increment step counters
        self.total_steps += 1
        self.steps_since_last_log += 1

    def save_metadata(
        self,
        location: str,
        robot_name: str,
        robot_type: str,
        evaluator_name: str,
        eval_name: Optional[str] = None,
    ):
        if self.data_saver:
            self.data_saver.save_metadata(
                location=location,
                robot_name=robot_name,
                robot_type=robot_type,
                evaluator_name=evaluator_name,
                eval_name=eval_name,
            )

    def log_success_rates(
        self,
        step,
        logging_prefix,
        episode_success,
    ):
        # save episode success
        if logging_prefix not in self.past_success_rates:
            self.past_success_rates[logging_prefix] = []
        self.past_success_rates[logging_prefix].append(float(episode_success))

        # success rates - round to 4 decimal places for precision
        recent_success_rate = round(
            np.mean(self.past_success_rates[logging_prefix][-20:]), 4
        )
        overall_success_rate = round(
            np.mean(self.past_success_rates[logging_prefix]), 4
        )

        # log
        to_log = {
            f"{logging_prefix}/episode_success": float(episode_success),
            f"{logging_prefix}/cumulative_num_success": sum(
                self.past_success_rates[logging_prefix]
            ),
            f"{logging_prefix}/recent_success_rate": recent_success_rate,
            f"{logging_prefix}/overall_success_rate": overall_success_rate,
        }
        return to_log
