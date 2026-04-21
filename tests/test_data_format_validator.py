"""Validate that a directory of logged data matches the robot_eval_logger format.

Usage:
    python tests/test_data_format_validator.py /path/to/storage_dir

Each sub-directory of <storage_dir> is treated as one evaluation run and must
contain a valid metadata.json and at least one traj_*.pkl.

Exit code 0 means all checks passed; non-zero means at least one error was found.
"""

import argparse
import json
import os
import pickle
import re
import sys
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (mirrors robot_eval_logger/typing/)
# ---------------------------------------------------------------------------

_LZ4_MAGIC = b"\x04\x22\x4d\x18"

_VALID_ROBOT_TYPES = {"franka", "widowx", "openarm"}
_VALID_CONTROL_MODES = {"joint_velocity", "joint_position", "end_effector"}

_REQUIRED_METADATA_FIELDS = {
    "eval_id",
    "robot_name",
    "robot_type",
    "control_mode",
    "action_frequency_hz",
    "time",
}

_NUMERIC_STEP_FIELDS = (
    "action",
    "joint_position",
    "joint_velocity",
    "end_effector_pose",
    "gripper",
    "joint_effort",
)

# Which step-level state field is required for each control mode
_CONTROL_MODE_REQUIRED_STATE = {
    "joint_velocity": "joint_velocity",
    "joint_position": "joint_position",
    "end_effector": "end_effector_pose",
}

_ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _err(msg: str) -> str:
    return f"  ERROR: {msg}"


def _warn(msg: str) -> str:
    return f"  WARN:  {msg}"


def _load_pickle(path: str):
    """Load a trajectory pickle, auto-detecting lz4 compression."""
    with open(path, "rb") as f:
        header = f.read(4)
        f.seek(0)
        if header == _LZ4_MAGIC:
            try:
                import lz4.frame
            except ImportError:
                raise ImportError(
                    "lz4 is required to read compressed pickles: pip install lz4"
                )
            return pickle.loads(lz4.frame.decompress(f.read()))
        return pickle.load(f)


def _infer_step_count(traj) -> Optional[int]:
    """Return the number of steps from the stacked action array, or episode_length."""
    action = getattr(traj, "action", None)
    if isinstance(action, np.ndarray) and action.ndim == 2:
        return action.shape[0]
    return getattr(traj, "episode_length", None)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_metadata(metadata_path: str) -> List[str]:
    """Return a list of error/warning strings for a metadata.json file."""
    issues = []

    if not os.path.isfile(metadata_path):
        return [_err(f"metadata.json not found at {metadata_path}")]

    try:
        with open(metadata_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [_err(f"metadata.json is not valid JSON: {e}")]

    if not isinstance(data, dict):
        return [_err("metadata.json must be a JSON object (dict)")]

    # Required fields presence
    for field in _REQUIRED_METADATA_FIELDS:
        if field not in data:
            issues.append(_err(f"metadata.json missing required field '{field}'"))

    # eval_id: integer
    if "eval_id" in data:
        if not isinstance(data["eval_id"], int):
            issues.append(
                _err(
                    f"metadata.json 'eval_id' must be an integer, "
                    f"got {type(data['eval_id']).__name__!r}: {data['eval_id']!r}"
                )
            )

    # robot_name: non-empty string
    if "robot_name" in data:
        if not isinstance(data["robot_name"], str) or not data["robot_name"].strip():
            issues.append(
                _err(
                    f"metadata.json 'robot_name' must be a non-empty string, "
                    f"got {data['robot_name']!r}"
                )
            )

    # robot_type: enum
    if "robot_type" in data:
        if data["robot_type"] not in _VALID_ROBOT_TYPES:
            issues.append(
                _err(
                    f"metadata.json 'robot_type' must be one of {sorted(_VALID_ROBOT_TYPES)}, "
                    f"got {data['robot_type']!r}. "
                    f"To add a new robot type, open a pull request."
                )
            )

    # control_mode: enum
    if "control_mode" in data:
        if data["control_mode"] not in _VALID_CONTROL_MODES:
            issues.append(
                _err(
                    f"metadata.json 'control_mode' must be one of "
                    f"{sorted(_VALID_CONTROL_MODES)}, got {data['control_mode']!r}. "
                    f"To add a new control mode, open a pull request."
                )
            )

    # action_frequency_hz: positive number
    if "action_frequency_hz" in data:
        val = data["action_frequency_hz"]
        if not isinstance(val, (int, float)):
            issues.append(
                _err(
                    f"metadata.json 'action_frequency_hz' must be a number, "
                    f"got {type(val).__name__!r}: {val!r}"
                )
            )
        elif val <= 0:
            issues.append(
                _err(f"metadata.json 'action_frequency_hz' must be positive, got {val}")
            )

    # time: ISO 8601 string
    if "time" in data:
        val = data["time"]
        if not isinstance(val, str):
            issues.append(
                _err(
                    f"metadata.json 'time' must be an ISO 8601 string, "
                    f"got {type(val).__name__!r}: {val!r}"
                )
            )
        elif not _ISO8601_RE.match(val):
            issues.append(
                _err(
                    f"metadata.json 'time' does not look like an ISO 8601 timestamp: {val!r}"
                )
            )

    # location: optional string
    if "location" in data and data["location"] is not None:
        if not isinstance(data["location"], str) or not data["location"].strip():
            issues.append(
                _err(
                    f"metadata.json 'location' must be a non-empty string or null, "
                    f"got {data['location']!r}"
                )
            )

    # evaluator_name, eval_name: optional strings
    for opt_field in ("evaluator_name", "eval_name"):
        if opt_field in data and data[opt_field] is not None:
            if not isinstance(data[opt_field], str):
                issues.append(
                    _err(
                        f"metadata.json '{opt_field}' must be a string or null, "
                        f"got {type(data[opt_field]).__name__!r}: {data[opt_field]!r}"
                    )
                )

    return issues


def validate_trajectory(
    traj_path: str, control_mode: Optional[str] = None
) -> List[str]:
    """Return a list of error/warning strings for a traj_*.pkl file.

    control_mode should be the validated value from the run's metadata.json
    (e.g. "joint_position"). When provided, the corresponding state field is
    checked as required.
    """
    issues = []
    fname = os.path.basename(traj_path)

    # Load
    try:
        traj = _load_pickle(traj_path)
    except Exception as e:
        return [_err(f"{fname}: failed to load pickle: {e}")]

    # language_command: required non-empty string
    _MISSING = object()
    lang = getattr(traj, "language_command", _MISSING)
    if lang is _MISSING:
        issues.append(_err(f"{fname}: missing required field 'language_command'"))
    elif not isinstance(lang, str):
        issues.append(
            _err(
                f"{fname}: 'language_command' must be a string, "
                f"got {type(lang).__name__!r}"
            )
        )
    elif not lang.strip():
        issues.append(_err(f"{fname}: 'language_command' is an empty string"))

    # success: required bool
    success = getattr(traj, "success", _MISSING)
    if success is _MISSING:
        issues.append(_err(f"{fname}: missing required field 'success'"))
    elif not isinstance(success, (bool, np.bool_)):
        issues.append(
            _err(
                f"{fname}: 'success' must be a bool, "
                f"got {type(success).__name__!r}: {success!r}"
            )
        )

    # episode_length: optional non-negative int
    ep_len = getattr(traj, "episode_length", None)
    if ep_len is not None:
        if not isinstance(ep_len, (int, np.integer)):
            issues.append(
                _err(
                    f"{fname}: 'episode_length' must be an int, "
                    f"got {type(ep_len).__name__!r}: {ep_len!r}"
                )
            )
        elif ep_len < 0:
            issues.append(
                _err(f"{fname}: 'episode_length' must be non-negative, got {ep_len}")
            )

    # duration_seconds: optional non-negative float
    dur = getattr(traj, "duration_seconds", None)
    if dur is not None:
        if not isinstance(dur, (int, float)):
            issues.append(
                _err(
                    f"{fname}: 'duration_seconds' must be a number, "
                    f"got {type(dur).__name__!r}: {dur!r}"
                )
            )
        elif dur < 0:
            issues.append(
                _err(f"{fname}: 'duration_seconds' must be non-negative, got {dur}")
            )

    # partial_success: optional float in [0, 1]
    ps = getattr(traj, "partial_success", None)
    if ps is not None:
        if not isinstance(ps, (int, float)):
            issues.append(
                _err(
                    f"{fname}: 'partial_success' must be a number, "
                    f"got {type(ps).__name__!r}: {ps!r}"
                )
            )
        elif not (0.0 <= float(ps) <= 1.0):
            issues.append(
                _err(f"{fname}: 'partial_success' must be in [0, 1], got {ps}")
            )

    # collection_time: optional ISO 8601 string
    ct = getattr(traj, "collection_time", None)
    if ct is not None:
        if not isinstance(ct, str):
            issues.append(
                _err(
                    f"{fname}: 'collection_time' must be a string, "
                    f"got {type(ct).__name__!r}: {ct!r}"
                )
            )
        elif not _ISO8601_RE.match(ct):
            issues.append(
                _err(
                    f"{fname}: 'collection_time' does not look like an ISO 8601 "
                    f"timestamp: {ct!r}"
                )
            )

    # Infer step count for cross-field consistency checks
    inferred_steps = _infer_step_count(traj)

    # --- Required step-level fields ---
    for required_field in ("obs", "action", "gripper"):
        if getattr(traj, required_field, None) is None:
            issues.append(
                _err(f"{fname}: missing required step-level field '{required_field}'")
            )

    if control_mode is not None:
        required_state = _CONTROL_MODE_REQUIRED_STATE.get(control_mode)
        if required_state and getattr(traj, required_state, None) is None:
            issues.append(
                _err(
                    f"{fname}: control_mode is '{control_mode}' but required field "
                    f"'{required_state}' is missing"
                )
            )

    # --- obs: must be a dict of camera_name -> (T, H, W, 3) uint8 ndarray ---
    obs = getattr(traj, "obs", None)
    if obs is not None:
        if not isinstance(obs, dict):
            issues.append(
                _err(
                    f"{fname}: 'obs' must be a dict mapping camera names to "
                    f"(T, H, W, 3) uint8 numpy arrays, got {type(obs).__name__!r}"
                )
            )
        else:
            for cam_name, arr in obs.items():
                if not isinstance(arr, np.ndarray):
                    issues.append(
                        _err(
                            f"{fname}: obs[{cam_name!r}] must be a numpy array of shape "
                            f"(T, H, W, 3), got {type(arr).__name__!r}"
                        )
                    )
                    continue
                if arr.ndim != 4 or arr.shape[3] != 3:
                    issues.append(
                        _err(
                            f"{fname}: obs[{cam_name!r}] must have shape (T, H, W, 3), "
                            f"got {arr.shape}"
                        )
                    )
                if arr.dtype != np.uint8:
                    issues.append(
                        _err(
                            f"{fname}: obs[{cam_name!r}] must have dtype uint8, "
                            f"got {arr.dtype}"
                        )
                    )
                if (
                    inferred_steps is not None
                    and arr.ndim == 4
                    and arr.shape[0] != inferred_steps
                ):
                    issues.append(
                        _err(
                            f"{fname}: obs[{cam_name!r}] has {arr.shape[0]} frames "
                            f"but episode has {inferred_steps} steps"
                        )
                    )

    # --- Numeric step-level fields: must be stacked 2-D float32 arrays ---
    for field_name in _NUMERIC_STEP_FIELDS:
        val = getattr(traj, field_name, None)
        if val is None:
            continue  # all step-level fields are optional

        if isinstance(val, list):
            issues.append(
                _err(
                    f"{fname}: '{field_name}' must be a stacked 2-D numpy array of "
                    f"shape (T, D) with dtype float32, but got a list. "
                    f"Use np.stack(your_list) to convert."
                )
            )
            continue

        if not isinstance(val, np.ndarray):
            issues.append(
                _err(
                    f"{fname}: '{field_name}' must be a numpy array, "
                    f"got {type(val).__name__!r}"
                )
            )
            continue

        if val.ndim != 2:
            issues.append(
                _err(
                    f"{fname}: '{field_name}' must be a 2-D array of shape (T, D), "
                    f"got shape {val.shape}"
                )
            )
        if val.dtype != np.float32:
            issues.append(
                _err(
                    f"{fname}: '{field_name}' must have dtype float32, got {val.dtype}"
                )
            )
        if val.ndim == 2:
            if inferred_steps is not None and val.shape[0] != inferred_steps:
                issues.append(
                    _err(
                        f"{fname}: '{field_name}' has {val.shape[0]} rows but "
                        f"episode has {inferred_steps} steps"
                    )
                )
            if field_name == "end_effector_pose" and val.shape[1] != 7:
                issues.append(
                    _err(
                        f"{fname}: 'end_effector_pose' must have 7 columns "
                        f"(xyz + quaternion), got {val.shape[1]}"
                    )
                )
            if field_name == "gripper" and val.shape[1] != 1:
                issues.append(
                    _warn(
                        f"{fname}: 'gripper' typically has 1 column, got {val.shape[1]}"
                    )
                )

    # episode_length consistency with inferred step count
    if ep_len is not None and inferred_steps is not None:
        if int(ep_len) != int(inferred_steps):
            issues.append(
                _err(
                    f"{fname}: 'episode_length' is {ep_len} but the step-level data "
                    f"has {inferred_steps} rows"
                )
            )

    return issues


def validate_run_dir(run_dir: str) -> Tuple[int, int]:
    """Validate one eval run directory. Returns (num_errors, num_warnings)."""
    dirname = os.path.basename(run_dir)
    errors = 0
    warnings = 0

    print(f"\n[Run] {dirname}")

    # metadata.json
    metadata_path = os.path.join(run_dir, "metadata.json")
    meta_issues = validate_metadata(metadata_path)
    if meta_issues:
        for issue in meta_issues:
            print(issue)
            if "ERROR" in issue:
                errors += 1
            else:
                warnings += 1
    else:
        print("  OK: metadata.json")

    # Extract control_mode for trajectory validation (best-effort; may be absent if metadata is bad)
    control_mode: Optional[str] = None
    try:
        with open(metadata_path) as f:
            _meta = json.load(f)
        cm = _meta.get("control_mode")
        if cm in _VALID_CONTROL_MODES:
            control_mode = cm
    except Exception:
        pass

    # traj_*.pkl files
    traj_files = sorted(
        f for f in os.listdir(run_dir) if re.match(r"^traj_\d+\.pkl$", f)
    )

    if not traj_files:
        print(_err("no traj_*.pkl files found in this run directory"))
        errors += 1
        return errors, warnings

    for traj_fname in traj_files:
        traj_issues = validate_trajectory(
            os.path.join(run_dir, traj_fname), control_mode=control_mode
        )
        if traj_issues:
            for issue in traj_issues:
                print(issue)
                if "ERROR" in issue:
                    errors += 1
                else:
                    warnings += 1
        else:
            print(f"  OK: {traj_fname}")

    # Check for gaps in trajectory numbering
    indices = [int(re.search(r"\d+", f).group()) for f in traj_files]
    if sorted(indices) != list(range(len(indices))):
        print(
            _warn(
                f"trajectory indices {sorted(indices)} are not contiguous starting "
                f"from 0 (expected {list(range(len(indices)))})"
            )
        )
        warnings += 1

    return errors, warnings


def validate_storage_dir(storage_dir: str) -> Tuple[int, int]:
    """Validate all eval run directories under storage_dir."""
    if not os.path.isdir(storage_dir):
        print(f"ERROR: '{storage_dir}' is not a directory or does not exist.")
        return 1, 0

    run_dirs = sorted(
        os.path.join(storage_dir, d)
        for d in os.listdir(storage_dir)
        if os.path.isdir(os.path.join(storage_dir, d))
    )

    if not run_dirs:
        print(f"ERROR: no run directories found inside '{storage_dir}'.")
        return 1, 0

    total_errors = 0
    total_warnings = 0
    for run_dir in run_dirs:
        errs, warns = validate_run_dir(run_dir)
        total_errors += errs
        total_warnings += warns

    return total_errors, total_warnings


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help="Path to the storage directory to validate.",
    )


def test_data_format(request):
    """pytest entry point: run with --data-dir=/path/to/storage_dir."""
    data_dir = request.config.getoption("--data-dir")
    if data_dir is None:
        import pytest

        pytest.skip("Pass --data-dir=<path> to run this test.")

    errors, warnings = validate_storage_dir(data_dir)

    if warnings:
        print(f"\n{warnings} warning(s) found (see WARN lines above).")
    assert errors == 0, (
        f"\n{errors} error(s) found in '{data_dir}'. "
        "See ERROR lines above for details."
    )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Validate that a directory matches the robot_eval_logger data format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "storage_dir",
        help="Root storage directory containing eval-run sub-directories.",
    )
    args = parser.parse_args()

    errors, warnings = validate_storage_dir(args.storage_dir)

    print()
    if errors == 0 and warnings == 0:
        print("All checks passed.")
    elif errors == 0:
        print(f"Passed with {warnings} warning(s).")
    else:
        print(
            f"FAILED: {errors} error(s), {warnings} warning(s). "
            "Fix the issues listed above."
        )

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
