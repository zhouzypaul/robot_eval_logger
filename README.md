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

## Getting Started
First, create the `EvalLogger` instance. When creating the `EvalLogger`, you can optionally pass in three functionalities this package provides:
- `wandb_logger`: log metrics to wandb. [Here](https://wandb.ai/rail-iterated-offline-rl/auto_eval_beta_launch/runs/test%20good%20_open_web_client_open_the_drawer_20250305_005411?nw=nwuserzhouzypaul) is an example of the logged metrics.
- `frames_visualizer`: make videos, film strips, and success information plotted against initial and final frames to visualize the evaluation. See [here](https://wandb.ai/rail-iterated-offline-rl/auto_eval_beta_launch/runs/test%20good%20_open_web_client_open_the_drawer_20250305_005411?nw=nwuserzhouzypaul) for an example.
- `data_saver`: save the logged data locally and / or upload to HuggingFace datasets.

If you don't need some of these functionalities, pass in `None` for the corresponding argument.

```python
from robot_eval_logger import (
    EvalLogger,
    FrameVisualizer,
    HuggingFaceStorage,
    LocalStorage,
    WandBLogger,
)

# wandb logger that logs metrics to wandb
wandb_config = WandBLogger.get_default_config()
wandb_logger = WandBLogger(
    wandb_config=wandb_config,
    variant={"exp_name": "testing_eval_logger"},
)

# frames visualizer that makes videos, film strips, and success information plotted against initial and final frames to visualize the evaluation
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
# Option 2: save data to HuggingFace
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

You can save the metadata for the evaluation:
```python
eval_logger.save_metadata(
    location="berkeley",
    robot_name="widowx_dummy",
    robot_type="widowx",
    evaluator_name="dummy_tester",
    eval_name="dummy_eval",
)
```

During each environment step, log transition-level data (observations, actions, proprio, etc.). The logger buffers these until the end of the episode.
```python
eval_logger.log_step(
    obs={"image_primary": image_hwc_uint8},
    action=action_array,
    proprio=proprio_array,
    joint_velocity=vel,  # optional
    joint_effort=effort,  # optional
)
```

After each evaluation episode, log episode-level metrics only. Step data from `log_step` is assembled automatically for storage and frame visualization.
```python
eval_logger.log_episode(
    i_episode=i_episode,
    logging_prefix="put the mushroom into the pot",
    episode_success=True,
    partial_success=0.0,  # optional episode-level kwargs also go here
)
```

You can also enable periodic evaluation throughput logging (steps per minute) by passing `log_step_stats_interval_minutes` when constructing `EvalLogger`; `log_step()` increments the internal step counters used for that feature.


## Example Usage

**Standalone dummy environment (no robot stack)** — `examples/dummy_eval.py` uses a small fake env that streams random images and proprio. Use this to try the logger after `pip install -e .` without any extra packages.
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
