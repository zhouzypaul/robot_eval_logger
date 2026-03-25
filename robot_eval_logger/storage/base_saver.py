import os
from dataclasses import dataclass
from typing import Optional, Union

from robot_eval_logger.typing import *
from robot_eval_logger.utils import make_eval_id_and_timestamp


@dataclass
class StorageConfig:
    """Configuration for storage optimizations."""

    # ~10-15x reduction in image size via JPEG encoding.
    # Slight CPU cost per image (~1-2 ms each); easily offset by smaller I/O.
    compress_images: bool = True

    # JPEG quality (1-100). Lower = smaller files but more compression artifacts.
    image_quality: int = 90

    # Stacks per-step List[np.ndarray] into contiguous (T, D) arrays.
    # Reduces pickle per-object overhead by ~5-10% and improves compressibility.
    # Negligible CPU cost.
    stack_arrays: bool = True

    # Wraps the pickle in lz4 frame compression.
    # ~20-40% additional size reduction with near-zero CPU overhead (lz4 is
    # designed for speed). Automatically detected on load.
    compress_pickle: bool = True

    # Uses ``pickle.HIGHEST_PROTOCOL`` (protocol 5 on Python 3.8+).
    # More efficient for large byte buffers (numpy arrays, JPEG blobs).
    # No measurable speed penalty; ~5% smaller pickles.
    use_highest_pickle_protocol: bool = True

    # Offloads ``save_episode`` to a single background thread so the eval loop
    # is never blocked by disk I/O or network uploads. Call ``flush()`` before
    # exiting to ensure all saves complete.
    async_saving: bool = True

    # (HuggingFaceStorage only) Queues per-episode uploads and commits them in
    # a single HF API call during ``flush()``. Dramatically reduces HTTP
    # overhead (one commit vs. N commits). Metadata is still uploaded eagerly.
    batch_hf_uploads: bool = False


class BaseSaver:
    def __init__(self, storage_dir, config: Optional[StorageConfig] = None):
        print(f"Eval data saving to {storage_dir}")
        self.storage_dir = storage_dir
        self.config = config or StorageConfig()

    def make_eval_id_and_timestamp(
        self, robot_type: str, eval_name: Optional[str] = None
    ):
        eval_id, time = make_eval_id_and_timestamp(
            robot_type=robot_type, eval_name=eval_name
        )
        self.eval_id = eval_id
        self.timestamp = time
        return eval_id, time

    def make_save_dir(self):
        save_dir = os.path.join(self.storage_dir, str(self.eval_id.id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"Specific run dir is {save_dir}")
        self.run_dir = save_dir
        return save_dir

    def save_metadata(
        self,
        location: str,
        robot_name: str,
        robot_type: str,
        control_mode: Union[str, ControlMode],
        action_frequency_hz: float,
        evaluator_name: Optional[str] = None,
        eval_name: Optional[str] = None,
    ):
        # must make eval_id first
        self.make_eval_id_and_timestamp(robot_type, eval_name)
        self.make_save_dir()
        # subclasses implement specific logic
        raise NotImplementedError

    def save_episode(self, i_episode: int, traj: TrajData):
        raise NotImplementedError

    def flush(self):
        """Wait for all pending async saves/uploads to complete.

        Subclasses that support ``async_saving`` must override this.
        """
        raise NotImplementedError
