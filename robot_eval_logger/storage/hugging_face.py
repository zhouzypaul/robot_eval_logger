"""
Eval data is always first saved locally, and then uploaded to HF.
"""
import logging
import os
import time
from typing import List, Optional, Tuple, Union

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from robot_eval_logger.storage.base_saver import StorageConfig
from robot_eval_logger.storage.local import LocalStorage
from robot_eval_logger.typing import *

logger = logging.getLogger(__name__)


class HuggingFaceStorage(LocalStorage):
    """Stores trajectory data and metadata to Hugging Face.

    Data is first written locally (benefiting from all :class:`LocalStorage`
    optimizations), then uploaded to a HF dataset repository.

    The Hugging Face dataset repo mirrors the same ``<eval_id>/`` tree under
    ``hf_dir_name`` (default ``"eval_data"``):

    - hf_dir_name
        - <eval_id>
            - metadata.json
            - traj_0.pkl
            - traj_1.pkl
            ...

    When ``StorageConfig.batch_hf_uploads`` is True (default), episode files are
    queued and committed to HuggingFace in a single API call during ``flush()``.
    This replaces N HTTP round-trips with one, dramatically reducing upload
    overhead for multi-episode evals.
    """

    def __init__(
        self,
        storage_dir: str,
        repo_id: str,
        hf_dir_name: str = "eval_data",
        config: Optional[StorageConfig] = None,
    ):
        """
        Args:
            storage_dir (str): Directory where the data is stored locally
            repo_id (str): Hugging Face repository ID
            hf_dir_name (str, optional): Directory name in the Hugging Face repository
            config (StorageConfig, optional): Storage optimizations configs
        """
        super().__init__(storage_dir, config)
        self.repo_id = repo_id
        self.hf_dir_name = hf_dir_name
        self.api = HfApi()
        # (local_path, repo_path, episode_index) tuples queued for batch upload
        self._pending_hf_uploads: List[Tuple[str, str, int]] = []

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

    def _batch_upload_to_hf(self, max_retries: int = 3) -> bool:
        """Commit all queued episode files in one HF API call.
        Falls back to individual uploads on failure.

        Returns:
            bool: True if upload was successful, False otherwise
        """
        if not self._pending_hf_uploads:
            return True

        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path)
            for local_path, repo_path, _ in self._pending_hf_uploads
        ]
        n = len(operations)
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Upload {n} episodes",
                )
                logger.info(f"Batch-uploaded {n} episodes to HuggingFace")
                return True
            except (
                HfHubHTTPError,
                RepositoryNotFoundError,
                ConnectionError,
                TimeoutError,
            ) as e:
                retry_count += 1
                wait_time = 2**retry_count
                logger.warning(
                    f"Batch upload failed (attempt {retry_count}/{max_retries}): {e}"
                )
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unexpected batch upload error: {e}")
                break

        logger.warning("Batch upload failed; falling back to individual uploads")
        all_ok = True
        for local_path, repo_path, i_ep in self._pending_hf_uploads:
            ok = self._upload_to_hf(local_path, repo_path, f"Upload episode {i_ep}")
            all_ok = all_ok and ok
        return all_ok

    def save_metadata(
        self,
        robot_name: str,
        robot_type: str,
        control_mode: Union[str, ControlMode],
        action_frequency_hz: float,
        location: Optional[str] = None,
        evaluator_name: Optional[str] = None,
        eval_name: Optional[str] = None,
    ):
        """Save metadata locally and upload to HuggingFace immediately."""
        metadata_path = super().save_metadata(
            location=location,
            robot_name=robot_name,
            robot_type=robot_type,
            control_mode=control_mode,
            action_frequency_hz=action_frequency_hz,
            evaluator_name=evaluator_name,
            eval_name=eval_name,
        )

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
                "Failed to upload metadata to HuggingFace, "
                f"but data was saved locally at: {metadata_path}"
            )

        return metadata_path

    def save_episode(self, i_episode: int, traj: TrajData):
        """Save ``traj`` locally, then upload (or queue) for HuggingFace.

        When ``config.batch_hf_uploads`` is True the upload is deferred to
        ``flush()``.
        """
        file_path = os.path.join(self.run_dir, f"traj_{i_episode}.pkl")
        path_in_repo = os.path.join(
            self.hf_dir_name, os.path.relpath(file_path, self.storage_dir)
        )

        if self.config.batch_hf_uploads:
            # save locally but don't upload to hf yet
            super().save_episode(i_episode, traj)
            self._pending_hf_uploads.append((file_path, path_in_repo, i_episode))
        else:
            # save locally and upload to hf
            if self.config.async_saving:
                future = self._executor.submit(
                    self._save_and_upload, file_path, traj, path_in_repo, i_episode
                )
                self._pending_futures.append(future)
            else:
                self._save_and_upload(file_path, traj, path_in_repo, i_episode)

        return file_path

    def _save_and_upload(
        self, file_path: str, traj: TrajData, path_in_repo: str, i_episode: int
    ):
        """Write locally then upload to HF (used when not batching)."""
        self._do_save(file_path, traj)
        ok = self._upload_to_hf(file_path, path_in_repo, f"Upload episode {i_episode}")
        if not ok:
            logger.warning(
                f"Failed to upload episode {i_episode} to HuggingFace, "
                f"but data was saved locally at: {file_path}"
            )

    def flush(self):
        """Finish all pending local saves, then batch-upload to HuggingFace."""
        super().flush()

        if self.config.batch_hf_uploads and self._pending_hf_uploads:
            self._batch_upload_to_hf()
            self._pending_hf_uploads.clear()
