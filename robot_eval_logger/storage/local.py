import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional, Union

from robot_eval_logger.storage.base_saver import BaseSaver, StorageConfig
from robot_eval_logger.typing import *

logger = logging.getLogger(__name__)


class LocalStorage(BaseSaver):
    """
    Stores the trajectory data and the metadata locally to disk
    Data is stored under eval_id/ folders:
    - storage_dir
        - <eval_id>
            - metadata.json
            - traj_0.pkl
            - traj_1.pkl
            ...
        - <eval_id>
    """

    def __init__(
        self,
        storage_dir: str,
        config: Optional[StorageConfig] = None,
    ):
        super().__init__(storage_dir, config)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending_futures: List[Future] = []
        if self.config.async_saving:
            self._executor = ThreadPoolExecutor(max_workers=1)

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
        """Create a MetaData object and save it as a json file."""
        self.make_eval_id_and_timestamp(robot_type, eval_name)
        self.make_save_dir()

        robot_type_enum = RobotType(robot_type)
        if isinstance(control_mode, str):
            control_mode_enum = ControlMode(control_mode)
        elif isinstance(control_mode, ControlMode):
            control_mode_enum = control_mode
        else:
            control_mode_enum = ControlMode(str(control_mode))

        metadata = MetaData(
            eval_id=self.eval_id,
            location=location,
            robot_name=robot_name,
            robot_type=robot_type_enum,
            control_mode=control_mode_enum,
            action_frequency_hz=float(action_frequency_hz),
            time=self.timestamp,
            evaluator_name=evaluator_name,
            eval_name=eval_name,
        )

        metadata_path = os.path.join(self.run_dir, "metadata.json")
        if os.path.exists(metadata_path):
            raise ValueError(f"metadata already exists at {metadata_path}")

        metadata.save(metadata_path)
        return metadata_path

    def save_episode(self, i_episode: int, traj: TrajData):
        """Pickle ``traj`` to ``traj_{i_episode}.pkl``.

        When ``config.async_saving`` is True the write is offloaded to a
        background thread so the eval loop is not blocked by disk I/O.
        """
        file_path = os.path.join(self.run_dir, f"traj_{i_episode}.pkl")
        if self.config.async_saving:
            future = self._executor.submit(self._do_save, file_path, traj)
            self._pending_futures.append(future)
        else:
            self._do_save(file_path, traj)
        return file_path

    def _do_save(self, file_path: str, traj: TrajData):
        """Perform the actual (possibly optimized) pickle write."""
        traj.save(
            file_path,
            compress_images=self.config.compress_images,
            image_quality=self.config.image_quality,
            stack_arrays=self.config.stack_arrays,
            compress_pickle=self.config.compress_pickle,
            use_highest_pickle_protocol=self.config.use_highest_pickle_protocol,
        )

    def flush(self):
        """Block until every queued background save has finished.

        Re-raises the first exception from any failed save.
        """
        errors = []
        for future in self._pending_futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Background save failed: {e}")
                errors.append(e)
        self._pending_futures.clear()
        if errors:
            raise errors[0]

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
