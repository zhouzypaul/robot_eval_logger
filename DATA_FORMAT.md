# Data Format Specification

This document describes the exact on-disk format that our evaluation logger produces. Please convert your dataset to match this format so it can be loaded by our tooling.

No knowledge of our codebase is required. You only need Python, NumPy, and lz4.

---

## Directory Layout

```
<storage_dir>/
└── <eval_id>/
    ├── metadata.json
    ├── traj_0.pkl
    ├── traj_1.pkl
    └── traj_2.pkl
    ...
```

- **`<storage_dir>`** — the root directory you choose.
- **`<eval_id>`** — a large positive integer (as a directory name) that uniquely identifies one evaluation run. All episodes from the same run go into the same `<eval_id>` directory.
- There is exactly **one `metadata.json`** per run directory, shared by all episodes.
- There is **one `traj_{i}.pkl`** per episode, numbered from zero (`traj_0.pkl`, `traj_1.pkl`, …).

---

## `metadata.json`

A UTF-8 JSON file containing a single object with the fields below.

| Field | Type | Required | Description |
|---|---|---|---|
| `eval_id` | integer | yes | Must match the directory name (large positive integer) |
| `robot_name` | string | yes | Human-readable name of the robot (e.g. `"franka_01"`) |
| `robot_type` | string (enum) | yes | Robot platform — see allowed values below |
| `control_mode` | string (enum) | yes | How the robot is commanded — see allowed values below |
| `action_frequency_hz` | float | yes | Policy/control rate in Hz (e.g. `10.0`). Must be positive. |
| `time` | string (ISO 8601) | yes | Run start time, e.g. `"2025-06-01T14:30:45.123456"` |
| `location` | string or null | no | Physical location of the evaluation (e.g. `"lab_a"`) |
| `evaluator_name` | string or null | no | Name of the human evaluator |
| `eval_name` | string or null | no | Human-readable name for this evaluation run |

### Allowed values for `robot_type`

| Value | Platform |
|---|---|
| `"franka"` | Franka Panda arm |
| `"widowx"` | WidowX arm |
| `"openarm"` | OpenArm |

> **Adding a new robot type** requires a pull request to this repository to add the value to the `RobotType` enum. Do not use unlisted values; they will be rejected by validation.

### Allowed values for `control_mode`

| Value | Meaning |
|---|---|
| `"joint_velocity"` | Commands are joint velocities |
| `"joint_position"` | Commands are joint positions |
| `"end_effector"` | Commands are end-effector poses |

> **Adding a new control mode** requires a pull request to this repository to add the value to the `ControlMode` enum. Do not use unlisted values; they will be rejected by validation.

### Example

```json
{
  "eval_id": 7823401928374650,
  "robot_name": "franka_01",
  "robot_type": "franka",
  "control_mode": "joint_position",
  "action_frequency_hz": 10.0,
  "time": "2025-06-01T14:30:45.123456",
  "location": "lab_a",
  "evaluator_name": "Alice",
  "eval_name": "pick_and_place_v2"
}
```

---

## `traj_{i}.pkl`

Each episode is stored as a **Python pickle file wrapped in lz4 frame compression**. The file contains a single Python object whose attributes hold episode and step data. The object's class name does not matter for loading; what matters is that the attributes listed below are present and have the correct types.

### Writing the file

```python
import pickle, lz4.frame

# ... build your episode object (see attributes below) ...

raw = pickle.dumps(episode, protocol=pickle.HIGHEST_PROTOCOL)
with open("traj_0.pkl", "wb") as f:
    f.write(lz4.frame.compress(raw))
```

### Reading the file

```python
import pickle, lz4.frame

with open("traj_0.pkl", "rb") as f:
    episode = pickle.loads(lz4.frame.decompress(f.read()))
```

---

### Attribute Summary

All attributes that must be present on the episode object, at a glance.

**Episode-level** (scalars describing the whole episode):

| Attribute | Type | Required |
|---|---|---|
| `language_command` | `str` | yes |
| `success` | `bool` | yes |
| `episode_length` | `int` or `None` | no |
| `duration_seconds` | `float` or `None` | no |
| `collection_time` | `str` (ISO 8601) or `None` | no |
| `partial_success` | `float` in [0, 1] or `None` | no |
| `language_feedback` | `str` or `None` | no |
| `policy_id` | `str` or `None` | no |

**Step-level** (arrays with one row per timestep `T`):

| Attribute | dtype | Shape | Required |
|---|---|---|---|
| `obs` | `dict[str, np.ndarray]` | each value: `(T, H, W, 3)` uint8 | **yes** |
| `action` | `float32` | `(T, D)` | **yes** |
| `joint_position` | `float32` | `(T, D)` | yes, if `control_mode` is `"joint_position"` |
| `joint_velocity` | `float32` | `(T, D)` | yes, if `control_mode` is `"joint_velocity"` |
| `end_effector_pose` | `float32` | `(T, 7)` | yes, if `control_mode` is `"end_effector"` |
| `gripper` | `float32` | `(T, 1)` | **yes** |
| `joint_effort` | `float32` | `(T, D)` | no |

`T` is the number of timesteps (`episode_length`), `D` is degrees of freedom, `H`/`W` are image height/width. The state field required by `control_mode` is determined by the run's `metadata.json`. Any step-level attribute that is present must match the dtype, shape, and length above. Additional arbitrary attributes are allowed.

---

### Episode-Level Attributes (details)

#### `language_command` *(required)*
`str` — Task instruction given to the robot, e.g. `"pick up the red block"`.

#### `success` *(required)*
`bool` — `True` if the episode succeeded, `False` otherwise.

#### `episode_length`
`int` or `None` — Number of logged timesteps. Should match the first dimension of all step-level arrays.

#### `duration_seconds`
`float` or `None` — Wall-clock duration of the episode in seconds.

#### `collection_time`
`str` (ISO 8601) or `None` — When the episode was logged, e.g. `"2025-06-01T14:31:00.000000"`.

#### `partial_success`
`float` or `None` — Partial success score in [0, 1].

#### `language_feedback`
`str` or `None` — Human evaluator's verbal feedback on the episode.

#### `policy_id`
`str` or `None` — Checkpoint or version identifier for the policy used.

---

### Step-Level Attributes (details)

#### `obs` — Camera Observations *(required)*

```
obs: dict[str, np.ndarray]
    keys:   camera name (any non-empty string, e.g. "image_primary", "image_wrist_left")
    values: np.ndarray   shape: (T, H, W, 3)   dtype: uint8   channel order: RGB
```

A dictionary mapping camera names to stacked image arrays. Each value is a `(T, H, W, 3)` uint8 array containing all frames for that camera across the episode. `H` and `W` can differ between cameras but must be consistent within a camera.

#### `action` — Command Sent to the Robot *(required)*

```
action: np.ndarray   shape: (T, D)   dtype: float32
```

One row per timestep. `D` is the number of degrees of freedom (e.g. `D=7` for a 7-DOF arm).

#### `joint_position` — Measured Joint Angles *(required when `control_mode` is `"joint_position"`)*

```
joint_position: np.ndarray   shape: (T, D)   dtype: float32
```

Units: radians.

#### `joint_velocity` — Measured Joint Velocities *(required when `control_mode` is `"joint_velocity"`)*

```
joint_velocity: np.ndarray   shape: (T, D)   dtype: float32
```

Units: radians per second.

#### `end_effector_pose` — End-Effector Pose *(required when `control_mode` is `"end_effector"`)*

```
end_effector_pose: np.ndarray   shape: (T, 7)   dtype: float32
```

Each row is `[x, y, z, qw, qx, qy, qz]` — Cartesian position followed by a unit quaternion.

#### `gripper` — Gripper State *(required)*

```
gripper: np.ndarray   shape: (T, 1)   dtype: float32
```

Convention: 0 = fully closed, 1 = fully open (exact range depends on robot platform).

#### `joint_effort` — Joint Torques *(optional)*

```
joint_effort: np.ndarray   shape: (T, D)   dtype: float32
```

May be omitted or set to `None` if not available.

---

## Minimal Working Example

The following self-contained script produces a compliant directory with one run and one episode. Use it as a template.

```python
import json, os, pickle
import lz4.frame
import numpy as np

# --- Parameters (adjust to match your data) ---
STORAGE_DIR = "./my_eval_data"
EVAL_ID     = 7823401928374650   # any large positive integer; used as directory name
T           = 50                  # number of timesteps
D           = 7                   # degrees of freedom
H, W        = 256, 256            # image height and width

# --- Create directory ---
run_dir = os.path.join(STORAGE_DIR, str(EVAL_ID))
os.makedirs(run_dir, exist_ok=True)

# --- Write metadata.json ---
metadata = {
    "eval_id":             EVAL_ID,
    "robot_name":          "franka_01",
    "robot_type":          "franka",           # "franka", "widowx", or "openarm"
    "control_mode":        "joint_position",   # "joint_velocity", "joint_position", or "end_effector"
    "action_frequency_hz": 10.0,
    "time":                "2025-06-01T14:30:45.123456",
    "location":            "lab_a",            # optional, can be omitted or null
    "evaluator_name":      "Alice",            # optional, can be omitted or null
    "eval_name":           "pick_and_place_v2" # optional, can be omitted or null
}
with open(os.path.join(run_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# --- Build episode object ---
class Episode:
    pass

ep = Episode()

# Episode-level fields
ep.language_command  = "pick up the red block"
ep.success           = True
ep.episode_length    = T
ep.duration_seconds  = 5.0
ep.collection_time   = "2025-06-01T14:31:00.000000"
ep.partial_success   = None   # or a float in [0, 1]
ep.language_feedback = None   # or a string
ep.policy_id         = None   # or a string

# Step-level fields: stacked arrays of shape (T, D), dtype float32
ep.action            = np.zeros((T, D), dtype=np.float32)
ep.joint_position    = np.zeros((T, D), dtype=np.float32)
ep.joint_velocity    = np.zeros((T, D), dtype=np.float32)
ep.end_effector_pose = np.zeros((T, 7), dtype=np.float32)
ep.gripper           = np.zeros((T, 1), dtype=np.float32)
ep.joint_effort      = None   # omit if not available

# Observations: dict from camera name -> stacked (T, H, W, 3) uint8 array
ep.obs = {
    "image_primary": np.zeros((T, H, W, 3), dtype=np.uint8),  # replace with your actual images
}

# --- Write traj_0.pkl (lz4-compressed pickle) ---
raw = pickle.dumps(ep, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(run_dir, "traj_0.pkl"), "wb") as f:
    f.write(lz4.frame.compress(raw))

print(f"Written to {run_dir}/")
```

For multiple episodes in the same run, write `traj_1.pkl`, `traj_2.pkl`, … into the same `run_dir`.

For multiple runs, create a new `<eval_id>` subdirectory for each and repeat the process.
