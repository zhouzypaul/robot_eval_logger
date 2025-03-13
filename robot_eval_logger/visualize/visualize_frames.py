from collections import deque
from typing import Tuple

import cv2
import moviepy.editor as mpy
import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class FrameVisualizer:
    """
    This class contains methods to nicely visualize the frames during a robot eval.
    """

    def __init__(
        self,
        video_fps: int = 10,
        video_frame_size: Tuple[int, int] = (128, 128),
        episode_viz_frame_interval: int = 10,
        periodic_log_initial_and_final_frames: bool = True,
        success_viz_every_n: int = 10,
    ):
        """
        Args:
            video_fps (int): fps of the video
            video_frame_size (Tuple[int, int]): frame size of the video
            episode_viz_frame_interval (int): create a sequence of frames,
                spaced by every n frames, to visualize an eval trajectory
            periodic_log_initial_and_final_frames (bool): whether to log the
                initial and final frames of the trajectory along with success predictions
            success_viz_every_n (int): log every n frames for periodic success predictions
                if set to True.
        """
        self.video_fps = video_fps
        self.video_frame_size = video_frame_size
        self.episode_viz_frame_interval = episode_viz_frame_interval
        self.success_viz_every_n = success_viz_every_n
        self.periodic_log_initial_and_final_frames = (
            periodic_log_initial_and_final_frames
        )

        self.past_frames = {}  # prefix --> list of frames

    def log_frames(
        self,
        step,
        logging_prefix,
        frames=None,
        success_rates=None,
    ):
        """
        visualize the frames as both a video and a sequence of spaced out frames.
        """
        if frames is None:
            return {}

        # save frames
        data = {
            "frames": (frames[0], frames[-1]) if len(frames) > 0 else None,
        }
        if logging_prefix not in self.past_frames:
            self.past_frames[logging_prefix] = deque(maxlen=self.success_viz_every_n)
        self.past_frames[logging_prefix].append(data)

        # frames to log
        to_log = {}
        if len(frames) > 0:
            # log every n frames
            every_n_frames = frames[:: self.episode_viz_frame_interval] + [frames[-1]]
            # Resize frames to ensure consistent display and avoid artifacts
            resized_frames = [
                cv2.resize(frame, self.video_frame_size) for frame in every_n_frames
            ]
            combined_frame = cv2.hconcat(resized_frames)
            to_log[f"{logging_prefix}/frames"] = wandb.Image(combined_frame)

            # log low quality video
            frame_size = self.video_frame_size
            tmp_filename = f"/tmp/{logging_prefix}_video.mp4"
            low_quality_frames = [cv2.resize(frame, frame_size) for frame in frames]
            # create the video clip
            clip = mpy.ImageSequenceClip(low_quality_frames, fps=self.video_fps)
            # write the video with libx264 encoding
            clip.write_videofile(tmp_filename, codec="libx264", preset="ultrafast")
            to_log[f"{logging_prefix}/video"] = wandb.Video(tmp_filename)

        # periodic logging of initial/final frames, along with success predictions
        if (
            (step + 1) % self.success_viz_every_n == 0
            and self.periodic_log_initial_and_final_frames
        ):
            assert len(self.past_frames[logging_prefix]) == self.success_viz_every_n

            frames = [data["frames"] for data in self.past_frames[logging_prefix]]
            frames = [frame for frame in frames if frame is not None]  # list of tuples
            img = self._plot_frames_with_success(
                frames=frames,
                past_success_rates=success_rates,
                step=step,
            )

            # log
            to_log[f"{logging_prefix}/initial_and_final_frames"] = wandb.Image(img)

            # pop off data already logged
            self.past_frames[logging_prefix].clear()

        return to_log

    def _plot_frames_with_success(self, frames, past_success_rates, step):
        """
        make a combined plot where the first two rows are the initial/final frames
        and the third row is the success predictions
        """
        n = len(frames)
        i_episode = range(step + 1 - n, step + 1)
        combined_frame = cv2.hconcat([cv2.vconcat(f) for f in frames])
        success_predictions = past_success_rates[-n:]

        fig, axs = plt.subplots(2, 1, figsize=(8, 4))
        canvas = FigureCanvas(fig)

        # frames
        axs[0].imshow(combined_frame)
        axs[0].set_axis_off()
        axs[0].set_title("Initial and final frames")

        # success predictions
        axs[1].plot(i_episode, success_predictions, marker="o")
        plt.xticks(i_episode)
        axs[0].set_title("Success predictions")

        plt.tight_layout()
        canvas.draw()
        out_image = np.array(canvas.renderer.buffer_rgba())
        return out_image

    def log_remaining_frames(self, final_step, success_rates=None):
        """
        Log any remaining frames in self.past_frames that haven't been logged due to
        not reaching the periodic logging threshold.

        Args:
            final_step (int): The final step number to use for logging
            success_rates (dict): Dictionary mapping logging_prefix to success rate lists

        Returns:
            dict: Dictionary of logged items to be passed to wandb
        """
        to_log = {}

        for logging_prefix, frames_data in self.past_frames.items():
            # Skip if no frames or already logged in the last periodic log
            if not frames_data or len(frames_data) == 0:
                continue

            # Extract frames and filter out None values
            frames = [data["frames"] for data in frames_data]
            frames = [frame for frame in frames if frame is not None]  # list of tuples

            if not frames:
                continue

            # Get corresponding success rates if available
            prefix_success_rates = None
            if success_rates is not None and logging_prefix in success_rates:
                prefix_success_rates = success_rates[logging_prefix]

            # Create visualization
            img = self._plot_frames_with_success(
                frames=frames,
                past_success_rates=prefix_success_rates
                if prefix_success_rates
                else [0] * len(frames),
                step=final_step,
            )

            # Log the image
            to_log[f"{logging_prefix}/initial_and_final_frames"] = wandb.Image(img)

        return to_log
