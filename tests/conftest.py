"""Pytest ``conftest.py``: shared fixtures for the ``tests/`` package.

How to run tests (from the repository root)::

    pytest
    pytest tests/test_storage_optimizations.py -v

If ``pytest`` fails with "unrecognized arguments: --cov=..." install dev deps
(``pytest-cov``) or disable project addopts for one run::

    python -m pytest -o addopts=
"""
import numpy as np
import pytest

from robot_eval_logger.typing.traj_data import TrajData


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_traj_data(rng):
    """A realistic TrajData with 3 cameras, 5 steps, and 7-DOF state."""
    n_steps = 5
    obs = {
        "image_primary": [
            rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(n_steps)
        ],
        "image_wrist_left": [
            rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(n_steps)
        ],
        "image_wrist_right": [
            rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(n_steps)
        ],
    }
    return TrajData(
        language_command="pick up the red block",
        success=True,
        episode_length=n_steps,
        duration_seconds=2.5,
        policy_id="test_policy",
        collection_time="2025-06-01T00:00:00",
        obs=obs,
        action=[rng.standard_normal(7).astype(np.float32) for _ in range(n_steps)],
        joint_position=[
            rng.standard_normal(7).astype(np.float32) for _ in range(n_steps)
        ],
        joint_velocity=[
            (0.01 * rng.standard_normal(7)).astype(np.float32) for _ in range(n_steps)
        ],
        end_effector_pose=[
            rng.standard_normal(7).astype(np.float32) for _ in range(n_steps)
        ],
        gripper=[rng.random(1).astype(np.float32) for _ in range(n_steps)],
        joint_effort=[
            (0.001 * rng.standard_normal(7)).astype(np.float32) for _ in range(n_steps)
        ],
    )
