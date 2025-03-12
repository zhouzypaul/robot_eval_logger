"""
An minimal example of how to use the EvalLogger
during an evaluation of a robot manipulator.

Usage: python examples/manipulator_eval.py --ip <robot_ip>
"""
import tempfile

from absl import app, flags
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.manipulator_env import ManipulatorEnv, StateEncoding
from manipulator_gym.utils.gym_wrappers import (
    ConvertState2Proprio,
    ResizeObsImageWrapper,
)

from robot_eval_logger import (
    EvalLogger,
    FrameVisualizer,
    HuggingFaceStorage,
    LocalStorage,
    WandBLogger,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("ip", "localhost", "IP address of the robot server.")
flags.DEFINE_string("text_cond", None, "Language prompt for the task.")

flags.DEFINE_integer("num_episodes", 7, "Number of episodes to evaluate.")
flags.DEFINE_integer("max_steps", 20, "Maximum number of steps per episode.")
flags.DEFINE_integer("log_every_n_frames", 3, "Log every n frames.")

flags.DEFINE_bool("debug", False, "Whether to debug or not.")
flags.DEFINE_string("exp_name", "", "Name of the experiment for wandb logging.")


def get_single_img(obs):
    img = obs["image_primary"]
    return img[-1] if img.ndim == 4 else img


def main(_):
    """
    create dummy policy for eval
    """
    dummy_null_policy = lambda obs, language_instruction: [0] * 7
    eval_policy = dummy_null_policy
    language_instruction = "dummy language"

    """
    Set up the logger
    """
    # wandb logger
    wandb_config = WandBLogger.get_default_config()
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.flag_values_dict(),
        debug=FLAGS.debug,
    )

    # frames visualizer
    frames_visualizer = FrameVisualizer(
        episode_viz_frame_interval=FLAGS.log_every_n_frames,
        success_viz_every_n=3,
        periodic_log_initial_and_final_frames=True,
    )

    # data saver
    # data_saver = LocalStorage(
    # storage_dir=tempfile.gettempdir(),
    # )
    data_saver = HuggingFaceStorage(
        storage_dir=tempfile.gettempdir(),
        repo_id="zhouzypaul/eval_logger",
    )

    # create the eval logger
    eval_logger = EvalLogger(
        wandb_logger=wandb_logger,
        frames_visualizer=frames_visualizer,
        data_saver=data_saver,
        log_step_stats_interval_minutes=None,  # frequent logging for testing
    )

    """
    create environment
    """
    manipulator_interface = ActionClientInterface(host=FLAGS.ip)

    def _create_env():
        env = ManipulatorEnv(
            manipulator_interface=manipulator_interface,
            state_encoding=StateEncoding.POS_EULER,
            use_wrist_cam=False,
        )
        env = ConvertState2Proprio(env)
        env = ResizeObsImageWrapper(
            env, resize_size={"image_primary": (256, 256), "image_wrist": (128, 128)}
        )

        return env

    env = _create_env()

    """
    Save eval metadata
    """
    eval_logger.save_metadata(
        location="berkeley",
        robot_name="widowx_dummy",
        robot_type="widowx",
        evaluator_name="dummy_tester",
        eval_name="dummy_eval",
    )

    """
    run eval rollouts
    """

    def eval_rollout():
        obs, info = env.reset(moving_time=5)
        frames_recorder = []
        infos = [info]

        for i in range(FLAGS.max_steps):
            actions = eval_policy(obs, language_instruction)
            print(f"Step {i} with action size of {len(actions)}")
            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            frames_recorder.append(get_single_img(obs))
            infos.append(info)

            eval_logger.log_step()

            if done or trunc:
                # trunc is because of robot failure
                break

        # return whether rollout is successful without robot failure
        execution_successful = not trunc
        eval_len = i

        # flatten the info
        infos = {k: [info[k] for info in infos] for k in infos[0].keys()}
        infos["eval_len"] = eval_len
        infos["frames"] = frames_recorder

        return obs, infos, execution_successful

    for i_episode in range(FLAGS.num_episodes):

        obs, eval_infos, eval_without_robot_error = eval_rollout()
        experienced_motor_failure = False

        # end of episode
        # success detection
        success = False
        print(f"Episode {i_episode} completed with success: {success}")

        # logging
        eval_logger.log_episode(
            i_episode=i_episode,
            logging_prefix=language_instruction,
            episode_success=success,
            frames_to_log=eval_infos["frames"],
            eval_rollout_steps=eval_infos["eval_len"],
            experienced_motor_failure=int(experienced_motor_failure),
        )

    # end of logger
    eval_logger.stop_time_logging()


if __name__ == "__main__":
    app.run(main)
