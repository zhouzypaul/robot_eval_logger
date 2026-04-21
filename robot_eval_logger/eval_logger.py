import time
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import wandb

from robot_eval_logger.storage import BaseSaver, LocalStorage
from robot_eval_logger.time import TimeLogger
from robot_eval_logger.typing import *


class EvalLogger:
    """
    Class to log robot evaluation metrics. This is the main interface to interact with the logger.
    This class offers the following main APIs:
        - log_step: log all metrics for a single step, users can call this at the end of every step
        - log_episode: log all metrics for an entire episode, users can call this at the end of every episode
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

        self.past_success_rates = {}  # prefix --> list of success data points
        self.metadata_saved = False

        # episode tracking
        self.current_episode = 0
        self._current_episode_steps = []
        self._episode_wall_start: Optional[float] = None
        self._step_input_check_passed = False

        # periodic step-throughput logging
        self._time_logger: Optional[TimeLogger] = None
        if (
            self.wandb_logger is not None
            and log_step_stats_interval_minutes is not None
        ):
            self._time_logger = TimeLogger(
                wandb_logger, log_step_stats_interval_minutes
            )
            self._time_logger.start()

    def log_episode(
        self,
        language_command: str,
        episode_success,
        *,
        viz_logging_prefix: Optional[str] = None,
        partial_success: Optional[float] = None,
        language_feedback: Optional[str] = None,
        policy_id: Optional[str] = None,
        **kwargs,
    ):
        """Log episode-level metrics at the end of an episode.

        Step-level data is assembled from prior log_step() calls. Run-level
        fields from :meth:`save_metadata` live in ``metadata.json`` only (not on each ``TrajData`` pickle).

        Episode index for wandb, storage (``traj_{i}.pkl``), and success-rate
        bookkeeping is ``self.current_episode`` (0-based), then incremented after
        each successful ``log_episode`` (so after a call, ``current_episode`` is
        the number of episodes logged).

        Args:
            language_command: Task instruction stored on ``TrajData`` and used as
                the default wandb / visualization key prefix unless overridden.
            episode_success: Binary success flag (``TrajData.success``).
            viz_logging_prefix: Optional wandb and frame-viz key prefix; defaults
                to ``language_command``.
            partial_success: ``TrajData.partial_success``.
            language_feedback: ``TrajData.language_feedback``.
            policy_id: ``TrajData.policy_id`` (checkpoint, run id, etc.).
            **kwargs: Extra episode-level keys for ``TrajData``.
        """
        i_episode = self.current_episode

        if viz_logging_prefix is None:
            viz_logging_prefix = language_command

        duration_seconds = (
            time.time() - self._episode_wall_start
            if self._episode_wall_start is not None
            else 0.0
        )

        traj = step_data_sequence_to_traj_data(self._current_episode_steps)
        frames_to_log = self._extract_frames(traj)

        traj_episode_fields = {
            k: v
            for k, v in (
                ("partial_success", partial_success),
                ("language_feedback", language_feedback),
            )
            if v is not None
        }
        traj_episode_fields["collection_time"] = datetime.now().isoformat()
        if policy_id is not None:
            traj_episode_fields["policy_id"] = policy_id

        # wandb stats and visualization
        if self.wandb_logger is not None:
            assert (
                viz_logging_prefix is not None
            ), "Doesn't support logging without a prefix currently"

            success_stats = self.log_success_rates(
                logging_prefix=viz_logging_prefix,
                episode_success=episode_success,
            )
            if self.frames_visualizer is not None:
                frames_viz = self.frames_visualizer.log_frames(
                    step=i_episode,
                    logging_prefix=viz_logging_prefix,
                    frames=frames_to_log,
                    success_rates=self.past_success_rates[viz_logging_prefix],
                )
            else:
                frames_viz = {}
            others = {f"{viz_logging_prefix}/{k}": v for k, v in kwargs.items()}
            to_log = {**frames_viz, **success_stats, **others}

            for k in to_log.keys():
                wandb.define_metric(f"{k}/*", step_metric="num_episode")
            self.wandb_logger.log({**to_log, "num_episode": i_episode})

        # save the trajectory data
        if self.data_saver:
            traj_data = TrajData(
                **traj.to_dict(),
                **kwargs,
                **traj_episode_fields,
                language_command=language_command,
                success=episode_success,
                episode_length=len(self._current_episode_steps),
                duration_seconds=duration_seconds,
            )
            self.data_saver.save_episode(i_episode=i_episode, traj=traj_data)

        self._reset_new_episode()
        self.current_episode += 1

        return to_log

    def _reset_new_episode(self):
        self._current_episode_steps = []
        self._episode_wall_start = None

    def _check_step_inputs(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        end_effector_pose: np.ndarray,
        gripper: np.ndarray,
        joint_effort: Optional[np.ndarray] = None,
    ):
        assert action.ndim == 1, "Action must be a 1D array, and not action chunks"
        assert joint_position.ndim == 1, "Joint position must be a 1D array"
        assert joint_velocity.ndim == 1, "Joint velocity must be a 1D array"
        assert end_effector_pose.ndim == 1, "End effector pose must be a 1D array"
        assert gripper.ndim == 1, "Gripper must be a 1D array"
        if joint_effort is not None:
            assert joint_effort.ndim == 1, "Joint effort must be a 1D array"
        self._step_input_check_passed = True

    def log_step(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        end_effector_pose: np.ndarray,
        gripper: np.ndarray,
        joint_effort: Optional[np.ndarray] = None,
    ):
        """Log one transition: multi-camera images, state, and action."""
        if not self._step_input_check_passed:
            self._check_step_inputs(
                obs,
                action,
                joint_position,
                joint_velocity,
                end_effector_pose,
                gripper,
                joint_effort,
            )

        if not self._current_episode_steps:
            self._episode_wall_start = time.time()
        if self._time_logger is not None:
            self._time_logger.record_step()
        self._current_episode_steps.append(
            StepData(
                obs=obs,
                action=action,
                joint_position=joint_position,
                joint_velocity=joint_velocity,
                end_effector_pose=end_effector_pose,
                gripper=gripper,
                joint_effort=joint_effort,
            )
        )

    def _extract_frames(self, traj: TrajData, obs_key="image_primary"):
        """Extract frames for visualization from a partial or full TrajData."""
        obs = getattr(traj, "obs", None)
        if obs is None:
            return None
        if isinstance(obs, dict):
            return obs.get(obs_key)
        return None

    def save_metadata(
        self,
        robot_name: str,
        robot_type: str,
        control_mode: Union[str, ControlMode],
        action_frequency_hz: float,
        location: Optional[str] = None,
        evaluator_name: Optional[str] = None,
        eval_name: Optional[str] = None,
    ):
        self.metadata_saved = True
        if self.data_saver:
            self.data_saver.save_metadata(
                location=location,
                robot_name=robot_name,
                robot_type=robot_type,
                control_mode=control_mode,
                action_frequency_hz=float(action_frequency_hz),
                evaluator_name=evaluator_name,
                eval_name=eval_name,
            )

    def log_success_rates(
        self,
        logging_prefix,
        episode_success,
    ):
        """visualizes success rate metrics"""
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

    def stop_time_logging(self):
        """Safely stop the time logging thread."""
        if self._time_logger is not None:
            self._time_logger.stop()

    def flush(self):
        """Wait for all pending async saves/uploads to complete.

        Safe to call multiple times.  Call this before exiting to ensure
        no trajectory data is lost when ``StorageConfig.async_saving``
        or ``StorageConfig.batch_hf_uploads`` are enabled.
        """
        if self.data_saver is not None:
            self.data_saver.flush()

    def __del__(self):
        """Clean up the time logging & saving thread when the logger is destroyed and log any remaining frames"""
        try:
            self._visualize_remaining_frames()
            self.flush()
            self.stop_time_logging()
        except Exception as e:
            print(f"Error in EvalLogger.__del__: {e}")
            pass  # ignore any errors during cleanup

    def _visualize_remaining_frames(self):
        """Log any remaining frames that haven't been logged due to not reaching the periodic threshold"""
        if self.frames_visualizer is None:
            return

        # Log the remaining frames (current_episode is next index; last logged is one less)
        final_episode_index = (
            self.current_episode - 1 if self.current_episode > 0 else 0
        )
        frames_viz = self.frames_visualizer.log_remaining_frames(
            final_step=final_episode_index, success_rates=self.past_success_rates
        )

        # Push to wandb if there's anything to log
        if frames_viz and self.wandb_logger is not None:
            self.wandb_logger.log({**frames_viz, "num_episode": final_episode_index})
