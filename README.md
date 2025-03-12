# robot_eval_logger
This package provides a logger for robot evaluations. The main functionalities are:
- **Flexible logging**: with a few lines of addition to your evaluation code, you can log any data you want
- **Visualization**: visualize the logged data (e.g. videos, success rates) easily through wandb ([weights & biases](https://wandb.ai)).
- **Data Storage**: store the logged data locally and / or upload to HuggingFace datasets


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
- `data_saver`: save the logged data locally and / or upload to HuggingFace datasets
If you don't need some / all of these, pass in `None` for the corresponding argument.

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

After each evaluation episode, log the episode data. You need to pass in the episode number, the prefix for the logging (e.g. language prompt), the success status, and the frames to log. You can also pass in any additional kwargs to log any data you want during an evaluation.
```python
kwargs = {}  # put any additional kwargs here
eval_logger.log_episode(
    i_episode=i_episode,
    logging_prefix="put the mushroom into the pot",
    episode_success=True,
    frames_to_log=[obs["image"] for obs in obs_list],
    **kwargs,
)
```

You can also log the evaluation throughput in terms of the number of evaluation steps taken per minute:
```python
eval_logger.log_step()
```


## Example Usage
See `examples/manipulator_eval.py` for an example of using the `EvalLogger` class in an evaluation loop on WidwoX robots. (For now, this script runs null actions on the robot, and you can use it to test the logger functionalities.)
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
