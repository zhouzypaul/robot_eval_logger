"""
Minimal EvalLogger demo with a fake environment (no robot / manipulator_gym).

Streams dummy images and proprio each step, logs with log_step / log_episode,
and saves trajectories under a temp directory by default.

Usage:
    python examples/dummy_eval.py
    python examples/dummy_eval.py --num_episodes 3 --max_steps 10 --debug
"""
import tempfile

import numpy as np
from absl import app, flags

from robot_eval_logger import (
    EvalLogger,
    FrameVisualizer,
    HuggingFaceStorage,
    LocalStorage,
    WandBLogger,
)

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 5, "Number of fake episodes.")
flags.DEFINE_integer("max_steps", 15, "Max environment steps per episode.")
flags.DEFINE_integer("log_every_n_frames", 3, "Frame interval for trajectory viz.")
flags.DEFINE_bool("debug", True, "If True, wandb runs in disabled mode (no login).")
flags.DEFINE_string("exp_name", "dummy_eval", "Experiment name for wandb config.")
flags.DEFINE_bool(
    "use_hf",
    False,
    "If True, upload trajectories to HuggingFace (requires login and repo_id).",
)
flags.DEFINE_string(
    "hf_repo_id",
    "zhouzypaul/eval_logger",
    "HF dataset repo when --use_hf is set.",
)


class DummyManipulatorEnv:
    """Tiny stand-in for a manipulator env: numpy obs only, no I/O."""

    def __init__(
        self,
        image_shape=(256, 256, 3),
        proprio_dim=7,
        seed=0,
    ):
        self.image_shape = image_shape
        self.proprio_dim = proprio_dim
        self.rng = np.random.default_rng(seed)
        self._step = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step = 0
        return self._make_obs(), {}

    def step(self, action):
        self._step += 1
        obs = self._make_obs()
        # Optional: end episode early sometimes to vary trajectory length
        done = bool(self.rng.random() < 0.05 and self._step > 3)
        trunc = False
        return obs, 0.0, done, trunc, {}

    def _make_obs(self):
        img = self.rng.integers(0, 256, size=self.image_shape, dtype=np.uint8)
        proprio = self.rng.standard_normal(self.proprio_dim).astype(np.float32)
        joint_velocity = 0.01 * self.rng.standard_normal(self.proprio_dim).astype(
            np.float32
        )
        joint_effort = 0.001 * self.rng.standard_normal(self.proprio_dim).astype(
            np.float32
        )
        return {
            "image_primary": img,
            "proprio": proprio,
            "joint_velocity": joint_velocity,
            "joint_effort": joint_effort,
        }


def main(_):
    language_instruction = "pick up the red block (dummy task)"

    wandb_config = WandBLogger.get_default_config()
    wandb_config.exp_descriptor = FLAGS.exp_name
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.flag_values_dict(),
        debug=FLAGS.debug,
    )

    frames_visualizer = FrameVisualizer(
        episode_viz_frame_interval=FLAGS.log_every_n_frames,
        success_viz_every_n=3,
        periodic_log_initial_and_final_frames=True,
    )

    storage_dir = tempfile.gettempdir()
    if FLAGS.use_hf:
        data_saver = HuggingFaceStorage(
            storage_dir=storage_dir,
            repo_id=FLAGS.hf_repo_id,
        )
    else:
        data_saver = LocalStorage(storage_dir=storage_dir)

    eval_logger = EvalLogger(
        wandb_logger=wandb_logger,
        frames_visualizer=frames_visualizer,
        data_saver=data_saver,
        log_step_stats_interval_minutes=None,
    )

    env = DummyManipulatorEnv(seed=42)

    eval_logger.save_metadata(
        location="local",
        robot_name="dummy_robot",
        robot_type="widowx",  # RobotType enum; synthetic env only
        evaluator_name="dummy_eval_script",
        eval_name="dummy_eval",
    )

    def dummy_policy(_obs, _lang):
        return np.zeros(7, dtype=np.float32)

    def eval_rollout():
        obs, _ = env.reset()
        last_i = 0
        for i in range(FLAGS.max_steps):
            action = dummy_policy(obs, language_instruction)
            print(f"Step {i}, action shape {action.shape}")
            obs, _reward, done, trunc, _info = env.step(action)

            eval_logger.log_step(
                obs={"image_primary": obs["image_primary"]},
                action=action,
                proprio=obs["proprio"],
                joint_velocity=obs["joint_velocity"],
                joint_effort=obs["joint_effort"],
            )

            last_i = i
            if done or trunc:
                break

        execution_ok = not trunc
        return obs, execution_ok, last_i

    for i_episode in range(FLAGS.num_episodes):
        _obs, _ok, eval_len = eval_rollout()
        experienced_motor_failure = False
        success = i_episode % 2 == 0  # alternate for demo success curves

        print(
            f"Episode {i_episode} done, success={success}, last step index={eval_len}"
        )
        eval_logger.log_episode(
            i_episode=i_episode,
            logging_prefix=language_instruction,
            episode_success=success,
            eval_rollout_steps=eval_len,
            experienced_motor_failure=int(experienced_motor_failure),
        )

    eval_logger.stop_time_logging()


if __name__ == "__main__":
    app.run(main)
