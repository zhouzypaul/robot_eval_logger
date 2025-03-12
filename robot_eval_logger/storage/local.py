import os
from typing import Dict, List, Optional

import numpy as np

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
        evaluator_name: str,
        eval_name: Optional[str] = None,
    ):
        """Create a MetaData object and save it as a json file"""
        # create eval_id and save_dir
        self.make_eval_id_and_timestamp(robot_type, eval_name)
        self.make_save_dir()

        # Create the MetaData object
        robot_type = RobotType(robot_type)
        metadata = MetaData(
            eval_id=self.eval_id,
            location=location,
            robot_name=robot_name,
            robot_type=RobotType(robot_type),
            time=self.timestamp,
            evaluator_name=evaluator_name,
        )

        metadata_path = os.path.join(self.run_dir, "metadata.json")
        if os.path.exists(metadata_path):
            raise ValueError(f"metadata already exists at {metadata_path}")

        metadata.save(metadata_path)
        return metadata_path

    def save_episode(
        self,
        i_episode: int,
        language_command: str,
        obs: Dict[str, np.ndarray],
        success: bool,
        action: Optional[List[np.ndarray]] = None,
        episode_length: Optional[int] = None,
        eval_duration: Optional[float] = None,
        proprio: Optional[List[np.ndarray]] = None,
        velocity: Optional[List[np.ndarray]] = None,
        effort: Optional[List[np.ndarray]] = None,
        partial_success: Optional[float] = None,
        language_feedback: Optional[str] = None,
        **kwargs,
    ):
        """Create a TrajData object and save it."""
        traj_data = TrajData(
            language_command=language_command,
            obs=obs,
            action=action,
            success=success,
            episode_length=episode_length,
            eval_duration=eval_duration,
            proprio=proprio,
            velocity=velocity,
            effort=effort,
            partial_success=partial_success,
            language_feedback=language_feedback,
            **kwargs,
        )

        # Define the file path where the TrajData will be saved
        file_path = os.path.join(self.run_dir, f"traj_{i_episode}.pkl")
        traj_data.save(file_path)

        return file_path
