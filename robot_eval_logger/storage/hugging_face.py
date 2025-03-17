"""
Eval data is always first saved locally, and then uploaded to hf
"""
import logging
import os
import time
from typing import Dict, List, Optional

from huggingface_hub import HfApi, HfFolder
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from robot_eval_logger.storage.local import LocalStorage
from robot_eval_logger.typing import *

# Set up logging
logger = logging.getLogger(__name__)


class HuggingFaceStorage(LocalStorage):
    """
    Stores the trajectory data and the metadata to Hugging Face
    Data is stored under eval_id/ folders in the Hugging Face repository.
    """

    def __init__(self, storage_dir: str, repo_id: str, hf_dir_name: str = "eval_data"):
        """
        Args:
            storage_dir (str): Directory where the data is stored locally
            repo_id (str): Hugging Face repository ID
            hf_dir_name (str, optional): Directory name in the Hugging Face repository
        """
        super().__init__(storage_dir)
        self.repo_id = repo_id
        self.hf_dir_name = hf_dir_name
        self.api = HfApi()

    def _upload_to_hf(
        self,
        file_path: str,
        path_in_repo: str,
        commit_message: str,
        max_retries: int = 3,
    ) -> bool:
        """Helper method to upload a file to Hugging Face with error handling and retries.

        Args:
            file_path: Local path to the file to upload
            path_in_repo: Path in the HF repository where the file should be stored
            commit_message: Commit message for the upload
            max_retries: Maximum number of retry attempts

        Returns:
            bool: True if upload was successful, False otherwise
        """
        retry_count = 0
        while retry_count < max_retries:
            try:
                with open(file_path, "rb") as f:
                    self.api.upload_file(
                        path_or_fileobj=f,
                        path_in_repo=path_in_repo,
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        commit_message=commit_message,
                    )
                return True
            except (
                HfHubHTTPError,
                RepositoryNotFoundError,
                ConnectionError,
                TimeoutError,
            ) as e:
                retry_count += 1
                wait_time = 2**retry_count  # Exponential backoff
                logger.warning(
                    f"HF upload failed (attempt {retry_count}/{max_retries}): {str(e)}"
                )
                logger.warning(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unexpected error during HF upload: {str(e)}")
                return False

        logger.error(
            f"Failed to upload {file_path} to Hugging Face after {max_retries} attempts"
        )
        return False

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
        path_in_repo = os.path.join(
            self.hf_dir_name, os.path.relpath(metadata_path, self.storage_dir)
        )
        success = self._upload_to_hf(
            file_path=metadata_path,
            path_in_repo=path_in_repo,
            commit_message="Upload metadata",
        )

        if not success:
            logger.warning(
                f"Failed to upload metadata to Hugging Face, but data was saved locally at: {metadata_path}"
            )

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

        # Upload to Hugging Face with error handling
        path_in_repo = os.path.join(
            self.hf_dir_name, os.path.relpath(traj_path, self.storage_dir)
        )
        upload_success = self._upload_to_hf(
            file_path=traj_path,
            path_in_repo=path_in_repo,
            commit_message=f"Upload episode {i_episode}",
        )

        if not upload_success:
            logger.warning(
                f"Failed to upload episode {i_episode} to Hugging Face, but data was saved locally at: {traj_path}"
            )

        return traj_path
