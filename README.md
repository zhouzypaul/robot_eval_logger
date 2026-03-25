# robot_eval_logger
This package provides utils for logging and saving deployment-style (e.g. evals, online RL) robot data. The main functionalities include:
- **Flexible logging**: with a few lines of addition to your evaluation code, you can log any data you want
- **Data Storage**: store the logged data locally and / or upload to HuggingFace datasets
- **Visualization**: visualize the logged data (e.g. videos, success rates) easily through wandb ([weights & biases](https://wandb.ai)).


## Installation
Install the package from source with
```bash
git clone git@github.com:zhouzypaul/robot_eval_logger.git
cd robot_eval_logger
pip install -e .
```

## Creating the logger
First, create the `EvalLogger` instance. When creating the `EvalLogger`, you can optionally pass in three functionalities this package provides:
- `wandb_logger`: log metrics to wandb. [Here](https://wandb.ai/rail-iterated-offline-rl/auto_eval_beta_launch/runs/test%20good%20_open_web_client_open_the_drawer_20250305_005411?nw=nwuserzhouzypaul) is an example of the logged metrics.
- `frames_visualizer`: make videos, film strips, and success information plotted against initial and final frames to visualize the evaluation. See [here](https://wandb.ai/rail-iterated-offline-rl/auto_eval_beta_launch/runs/test%20good%20_open_web_client_open_the_drawer_20250305_005411?nw=nwuserzhouzypaul) for an example.
- `data_saver`: save the logged data locally and / or upload to HuggingFace datasets.

**NOTE**: If you don't need some of these functionalities, pass in `None` for the corresponding argument. For instance, if you only want to save the evaluation/deployment data, you only need to pass in `data_saver`, and set `wandb_logger` and `frames_visualizer` to `None`.

```python
from robot_eval_logger import (
    EvalLogger,
    FrameVisualizer,
    HuggingFaceStorage,
    LocalStorage,
    WandBLogger,
)

# [Optional] wandb logger that logs metrics to wandb
wandb_config = WandBLogger.get_default_config()
wandb_logger = WandBLogger(
    wandb_config=wandb_config,
    variant={"exp_name": "testing_eval_logger"},
)

# [Optional] frames visualizer that makes videos, film strips, and success information plotted against initial and final frames to visualize the evaluation
frames_visualizer = FrameVisualizer(
    episode_viz_frame_interval=10,  # when creating a film strip of the eval trajectory, visualize a frame every 10 frames
    success_viz_every_n=3,  # for every 3 episodes, visualize success information alongside initial and final frames
    periodic_log_initial_and_final_frames=True,  # visualize initial and final frames alongside success information
)

# data saver
# Option 1: save data locally
data_saver = LocalStorage(
    storage_dir='path/to/your/storage',
)
# [Recommended] Option 2: save data to HuggingFace
data_saver = HuggingFaceStorage(
    storage_dir='path/to/your/storage',
    repo_id="HF_USERNAME/eval_logger",
    hf_dir_name="eval_data",
)

# create the eval logger
# You can pass in None for any of the arguments if you don't need some / all of the functionalities
eval_logger = EvalLogger(
    wandb_logger=wandb_logger,
    frames_visualizer=frames_visualizer,
    data_saver=data_saver,
)
```

## Using the logger

There is three things you need to do to use the logger: (1) save the metadata for the current evaluation run, (2) call `log_step` after each interation with the environment, and (3) call `log_episode` at the end of each episode.

### (1) Saving the metadata
You need to save the metadata for the current evaluation run before doing anything else with the logger. The metadata includes the location, robot name, robot type, evaluator name, eval name, control mode, and action frequency. It is saved as a json file.

```python
from robot_eval_logger import ControlMode

eval_logger.save_metadata(
    location="berkeley",
    robot_name="widowx_dummy",
    robot_type="widowx",
    control_mode=ControlMode.JOINT_POSITION,  # or "joint_velocity", "end_effector"
    evaluator_name="dummy_tester",
    eval_name="dummy_eval",  # optional
    action_frequency_hz=10.0,  # required: control / policy command rate (Hz)
)
```

#### Robot types (`RobotType`)

You need to log the robot type for each evaluation run. The current supported robot types are in `robot_eval_logger/typing/eval_metadata.py`:


| Member | String value |
|--------|----------------|
| `RobotType.FRANKA` | `"franka"` |
| `RobotType.WIDOWX` | `"widowx"` |

If you are using a different robot, please make sure to add a new member to the `RobotType` enum, and make a Pull Request to this package.

### (2) Logging the step

**Per timestep** (`log_step`): multi-camera images (`obs` dict: e.g. base + wrists), joint position, joint velocity, end-effector pose (flat vector), gripper state, commanded action, and optional joint effort/torque.

```python
eval_logger.log_step(
    obs={
        "image_primary": base_cam_hwc_uint8,
        "image_wrist_left": wrist_l_hwc_uint8,
        "image_wrist_right": wrist_r_hwc_uint8,
    },
    action=action_array,
    joint_position=joint_pos,
    joint_velocity=joint_vel,
    end_effector_pose=ee_pose_seven,  # e.g. xyz + quat; shape is up to you
    gripper=gripper_array,  # e.g. shape (1,) for width or openness
    joint_effort=torque,  # optional
)
```


You can also enable periodic evaluation throughput logging (steps per minute) by passing `log_step_stats_interval_minutes` when constructing `EvalLogger`; `log_step()` increments the internal step counters used for that feature.

### (3) Logging the episode

Make sure to call `log_episode` at the end of each episode. The logger depends on this to track the different episodes and reset internal states.

**Per episode** (`log_episode`): episode index for `traj_{i}.pkl` and wandb is tracked internally (`current_episode`, incremented after each call); **`language_command`** (stored on `TrajData` and, by default, used as the wandb key prefix), binary **success**, optional **`viz_logging_prefix`** (overrides the wandb/frame-viz prefix without changing saved `language_command`), optional **policy id**, plus any extra **kwargs** (e.g. `partial_success`). **Collection time** (`datetime.now().isoformat()`) and **policy id** are stored on each `TrajData` pickle but are **not** sent to wandb. **Wandb** gets only success-rate metrics, frame visualizations, and kwargs (prefixed with the viz prefix, usually the language command). **Run-level** fields from `save_metadata` stay in `metadata.json` only. Pair `traj_*.pkl` with `metadata.json` in the eval directory when analyzing data.

Run-level context is stored in `metadata.json` via `save_metadata`: location, robot name, robot type, evaluator, eval name, control mode, **action frequency (Hz, required)**, run start time.

```python
eval_logger.log_episode(
    language_command="put the mushroom into the pot",
    episode_success=False,
    policy_id="pi05_checkpoint_12345",
    partial_success=0.6,
)
```

## Example Usage

**Standalone dummy environment (no robot stack)** — `examples/dummy_eval.py` uses a small fake env (multi-camera images, joints, EE pose, gripper). Use it after `pip install -e .` without extra robot packages.
```bash
python examples/dummy_eval.py
# wandb disabled by default (--debug defaults to True); real logging: --nodebug
# Local disk only by default; HuggingFace upload: add --use_hf (see script flags)
```

**Real manipulator** — `examples/manipulator_eval.py` wires `EvalLogger` into a WidowX-style loop via `manipulator_gym` (separate install). It sends null actions for testing.
```bash
python examples/manipulator_eval.py --ip <robot_ip>
```

### Uploading data to HF
You must be authenticated with a write token with
```bash
huggingface-cli login
```

## Contributing
To enable code checks and auto-formatting, please install pre-commit hooks (run this in the root directory):
```
pre-commit install
```
The hooks should now run before every commit. If files are modified during the checks, you'll need to re-stage them and commit again.
