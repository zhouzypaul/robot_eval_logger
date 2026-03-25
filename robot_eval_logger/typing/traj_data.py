import io
import pickle
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import numpy as np

_LZ4_MAGIC = b"\x04\x22\x4d\x18"

_NUMERIC_STEP_FIELDS = (
    "action",
    "joint_position",
    "joint_velocity",
    "end_effector_pose",
    "gripper",
    "joint_effort",
)


@dataclass
class StepData:
    """Single-timestep data for EvalLogger.log_step().

    obs maps camera names to one image each (e.g. base + wrists).
    end_effector_pose is a flat vector (e.g. xyz + rotation as your stack uses).
    gripper is typically a 1-vector (width, openness, or binary).
    """

    obs: Dict[str, np.ndarray]
    action: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    end_effector_pose: np.ndarray
    gripper: np.ndarray
    joint_effort: Optional[np.ndarray] = None


@dataclass
class TrajData:
    """One episode written to ``traj_{i}.pkl`` under an eval run directory.

    Meta-level facts (robot platform, location, control mode, action frequency,
    eval name, …) are stored once in ``metadata.json`` in that directory—do not
    expect them to be repeated on every trajectory file. Join trajectories to the
    run using the shared folder / ``eval_id``.

    Typical attributes (episode-level):
        language_command, success, episode_length,
        collection_time (``datetime.now().isoformat()``, set at ``log_episode``),
        policy_id (optional), partial_success, language_feedback,
        duration_seconds (wall time for the episode, set by :class:`EvalLogger`), …

    Step-level (lists, one entry per logged timestep):
        obs (camera -> list of images), action, joint_position, joint_velocity,
        end_effector_pose, gripper, joint_effort

    Extra keys are accepted via ``__init__(**kwargs)`` (e.g. legacy pickles).
    """

    language_command: str
    success: bool
    episode_length: Optional[int] = None
    duration_seconds: Optional[float] = None
    partial_success: Optional[float] = None
    language_feedback: Optional[str] = None
    policy_id: Optional[str] = None
    collection_time: Optional[str] = None

    obs: Optional[Dict[str, List[np.ndarray]]] = None
    action: Optional[List[np.ndarray]] = None
    joint_position: Optional[List[np.ndarray]] = None
    joint_velocity: Optional[List[np.ndarray]] = None
    end_effector_pose: Optional[List[np.ndarray]] = None
    gripper: Optional[List[np.ndarray]] = None
    joint_effort: Optional[List[np.ndarray]] = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(
        self,
        file_path: str,
        *,
        compress_images: bool = False,
        image_quality: int = 90,
        stack_arrays: bool = False,
        compress_pickle: bool = False,
        use_highest_pickle_protocol: bool = False,
    ) -> None:
        """Serialize this trajectory to *file_path* with optional optimizations"""
        save_obj = self
        if compress_images or stack_arrays:
            save_obj = self._prepare_for_save(
                compress_images, image_quality, stack_arrays
            )

        protocol = (
            pickle.HIGHEST_PROTOCOL
            if use_highest_pickle_protocol
            else pickle.DEFAULT_PROTOCOL
        )

        if compress_pickle:
            import lz4.frame

            data = pickle.dumps(save_obj, protocol=protocol)
            with open(file_path, "wb") as f:
                f.write(lz4.frame.compress(data))
        else:
            with open(file_path, "wb") as f:
                pickle.dump(save_obj, f, protocol=protocol)

    def _prepare_for_save(
        self, compress_images: bool, image_quality: int, stack_arrays: bool
    ) -> "TrajData":
        """Return a lightweight copy with images JPEG-encoded and/or arrays stacked."""
        data = self.to_dict()

        if compress_images and "obs" in data and data["obs"] is not None:
            data["obs"] = _encode_obs_images(data["obs"], image_quality)

        if stack_arrays:
            for name in _NUMERIC_STEP_FIELDS:
                val = data.get(name)
                if isinstance(val, list) and len(val) > 0:
                    try:
                        data[name] = np.stack(val)
                    except (ValueError, TypeError):
                        pass

        return TrajData(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dict__.keys()}

    @staticmethod
    def load(file_path: str) -> "TrajData":
        """Load a trajectory pickle, auto-detecting lz4 compression."""
        with open(file_path, "rb") as f:
            header = f.read(4)
            f.seek(0)
            if header == _LZ4_MAGIC:
                import lz4.frame

                return pickle.loads(lz4.frame.decompress(f.read()))
            return pickle.load(f)

    @staticmethod
    def decode_images(
        obs: Dict[str, list],
    ) -> Dict[str, List[np.ndarray]]:
        """Decode JPEG-encoded obs images back to numpy arrays.

        Accepts a mixed dict — entries that are already ``np.ndarray`` are
        passed through unchanged, so this is safe to call unconditionally.
        """
        from PIL import Image

        decoded: Dict[str, List[np.ndarray]] = {}
        for cam, img_list in obs.items():
            decoded[cam] = [
                np.asarray(Image.open(io.BytesIO(img)))
                if isinstance(img, bytes)
                else img
                for img in img_list
            ]
        return decoded


def _encode_obs_images(obs: Dict[str, list], quality: int = 85) -> Dict[str, list]:
    """JPEG-encode each image array in *obs*, returning ``bytes`` blobs."""
    from PIL import Image

    encoded: Dict[str, list] = {}
    for cam_name, img_list in obs.items():
        out = []
        for img in img_list:
            if isinstance(img, np.ndarray):
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="JPEG", quality=quality)
                out.append(buf.getvalue())
            else:
                out.append(img)
        encoded[cam_name] = out
    return encoded


def _transpose_dicts_per_timestep(dicts: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """``[{"a": x1, "b": y1}, {"a": x2, "b": y2}]`` → ``{"a": [x1, x2], "b": [y1, y2]}``."""
    all_keys = {k for d in dicts for k in d}
    return {k: [d.get(k) for d in dicts] for k in all_keys}


def step_data_sequence_to_traj_data(steps: List[StepData]) -> TrajData:
    """Aggregate :class:`StepData` rows into a :class:`TrajData` with step-level fields only.

    Episode-level fields (``language_command``, ``success``, …) must be set by the caller
    before saving.
    """
    if not steps:
        return TrajData()
    step_kw: Dict[str, Any] = {}
    for field in fields(StepData):
        values = [getattr(step, field.name) for step in steps]
        if not any(v is not None for v in values):
            continue
        if isinstance(values[0], dict):
            step_kw[field.name] = _transpose_dicts_per_timestep(values)
        else:
            step_kw[field.name] = values
    return TrajData(**step_kw)


def main():
    # for testing TrajData class
    obs_images = {
        f"camera{i}": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        for i in range(1, 4)
    }
    traj_data = TrajData(
        language_command="Test command",
        obs={k: [v] for k, v in obs_images.items()},
        action=[np.array([0.1, 0.2])],
        joint_position=[np.zeros(7)],
        joint_velocity=[np.zeros(7)],
        end_effector_pose=[np.zeros(7)],
        gripper=[np.array([0.0])],
        success=True,
        partial_success=0.8,
        episode_length=10,
        duration_seconds=5.0,
        policy_id="test_policy",
        collection_time="2025-01-01T00:00:00+00:00",
        some_other_field="just a test",
    )
    traj_data.save("/tmp/traj_data.pkl")
    TrajData.load("/tmp/traj_data.pkl")


if __name__ == "__main__":
    main()
