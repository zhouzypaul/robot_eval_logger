"""
Eval data is always first saved locally, and then uploaded to hf
"""
import os

from huggingface_hub import HfApi, HfFolder

from robot_eval_logger.storage.local import LocalStorage
from robot_eval_logger.typing import *


class HuggingFaceStorage(LocalStorage):
    """
    Stores the trajectory data and the metadata to Hugging Face
    Data is stored under eval_id/ folders in the Hugging Face repository.
    """

    def __init__(self, storage_dir: str, repo_id: str, hf_dir_name: str = "eval_data"):
        super().__init__(storage_dir)
        self.repo_id = repo_id
        self.hf_dir_name = hf_dir_name
        self.api = HfApi()

    def save_metadata(
        self,
        location: str,
        robot_name: str,
        robot_type: str,
        evaluator_name: str,
        eval_name: Optional[str] = None,
    ):
        """Create a MetaData object and save it to Hugging Face"""
        metadata_path = super().save_metadata(
            location, robot_name, robot_type, evaluator_name, eval_name
        )

        # Save metadata to Hugging Face
        self.api.upload_file(
            path_or_fileobj=open(metadata_path, "rb"),
            path_in_repo=os.path.join(
                self.hf_dir_name, os.path.basename(metadata_path)
            ),
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Upload metadata",
        )

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
        """Create a TrajData object and save it to Hugging Face."""
        traj_path = super().save_episode(
            i_episode,
            language_command,
            obs,
            success,
            action,
            episode_length,
            eval_duration,
            proprio,
            velocity,
            effort,
            partial_success,
            language_feedback,
            **kwargs,
        )

        self.api.upload_file(
            path_or_fileobj=open(traj_path, "rb"),
            path_in_repo=os.path.join(self.hf_dir_name, os.path.basename(traj_path)),
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Upload episode {i_episode}",
        )
