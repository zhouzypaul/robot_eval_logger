import os
from typing import Optional, Union

from robot_eval_logger.storage.base_saver import BaseSaver
from robot_eval_logger.typing import *


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
        """Create a MetaData object and save it as a json file"""
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
        """Pickle ``traj`` to ``traj_{i_episode}.pkl``."""
        file_path = os.path.join(self.run_dir, f"traj_{i_episode}.pkl")
        traj.save(file_path)
        return file_path
