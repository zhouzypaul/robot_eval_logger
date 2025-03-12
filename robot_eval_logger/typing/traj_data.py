import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrajData:
    language_command: str
    obs: Dict[str, np.ndarray]  # mapping different cameras to images
    success: bool
    action: Optional[List[np.ndarray]] = None
    episode_length: Optional[int] = None
    eval_duration: Optional[float] = None
    proprio: Optional[List[np.ndarray]] = None
    velocity: Optional[List[np.ndarray]] = None
    effort: Optional[List[np.ndarray]] = None
    partial_success: Optional[
        float
    ] = None  # 0 to 1, for partial credit towards success
    language_feedback: Optional[str] = None

    def __init__(self, **kwargs):
        """store arbitrary data"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self, file_path: str) -> None:
        """Save the TrajData instance to a file."""
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path: str) -> "TrajData":
        """Load a TrajData instance from a file."""
        with open(file_path, "rb") as file:
            return pickle.load(file)


def main():
    """for testing TrajData"""
    # Create random 256x256 pixel images for obs
    obs_images = {
        f"camera{i}": np.random.randint(
            0, 256, (256, 256, 3), dtype=np.uint8
        )  # Random RGB images
        for i in range(1, 4)  # Create images for 3 cameras
    }

    # Create an instance of TrajData
    traj_data = TrajData(
        language_command="Test command",
        obs=obs_images,
        action=[np.array([0.1, 0.2])],
        success=True,
        partial_success=0.8,
        episode_length=10,
        eval_duration=5.0,
        some_other_field="just a test",
    )

    # Save the instance to a file
    traj_data.save("/tmp/traj_data.pkl")

    # Load the instance from the file
    loaded_traj_data = TrajData.load("/tmp/traj_data.pkl")


if __name__ == "__main__":
    main()
