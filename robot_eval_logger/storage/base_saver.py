import os

from robot_eval_logger.typing import *
from robot_eval_logger.utils import make_eval_id_and_timestamp


class BaseSaver:
    def __init__(self, storage_dir):
        print(f"Eval data saving to {storage_dir}")
        self.storage_dir = storage_dir

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
        evaluator_name: str,
        eval_name: Optional[str] = None,
    ):
        # must make eval_id first
        self.make_eval_id_and_timestamp(robot_type, eval_name)
        self.make_save_dir()
        # subclasses implement specific logic
        raise NotImplementedError

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
        raise NotImplementedError
