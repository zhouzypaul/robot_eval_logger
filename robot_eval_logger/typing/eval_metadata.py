import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class RobotType(Enum):
    FRANKA = "franka"
    OPENARM = "openarm"
    WIDOWX = "widowx"


class ControlMode(str, Enum):
    """How the robot is commanded for this eval / deployment run (run-level metadata)."""

    JOINT_VELOCITY = "joint_velocity"
    JOINT_POSITION = "joint_position"
    END_EFFECTOR = "end_effector"


class TimeStamp:
    def __init__(self):
        self.timestamp = datetime.now()

    def __str__(self):
        # Returns the timestamp in ISO 8601 format
        return self.timestamp.isoformat()

    def formatted(self, format_string: str):
        # a method to format the timestamp
        return self.timestamp.strftime(format_string)


@dataclass
class EvalID:
    id: str

    @classmethod
    def create(cls, time: TimeStamp, robot_type: RobotType, custom_name: str = None):
        """Generates a hash for EvalID based on TimeStamp, robot_type,
        and optional custom_name."""
        hash_input = f"{time}{robot_type}{custom_name or ''}"
        return cls(id=abs(hash(hash_input)))


@dataclass
class MetaData:
    eval_id: EvalID
    location: str
    robot_name: str
    robot_type: RobotType
    control_mode: ControlMode
    action_frequency_hz: float
    time: TimeStamp
    evaluator_name: Optional[str] = None
    eval_name: Optional[str] = None

    def save(self, file_path: str):
        """Saves the MetaData instance to a JSON file."""
        # Convert eval_id to a serializable format
        data_to_save = {
            "eval_id": self.eval_id.id,  # Save only the id
            "location": self.location,
            "robot_name": self.robot_name,
            "robot_type": self.robot_type.value,  # Save robot_type as string
            "time": self.time.timestamp.isoformat(),  # Save timestamp as string
            "evaluator_name": self.evaluator_name,
            "eval_name": self.eval_name,
            "control_mode": self.control_mode.value,
            "action_frequency_hz": self.action_frequency_hz,
        }
        with open(file_path, "w") as json_file:
            json.dump(data_to_save, json_file)

    @classmethod
    def load(cls, file_path: str):
        """Loads MetaData from a JSON file."""
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
            # Reconstruct the EvalID and TimeStamp objects
            eval_id = EvalID(id=data["eval_id"])
            robot_type = RobotType(data["robot_type"])  # Convert back to RobotType
            time = TimeStamp()
            time.timestamp = datetime.fromisoformat(
                data["time"]
            )  # Convert back to datetime
            if "control_mode" not in data:
                raise ValueError(
                    "metadata.json is missing required field 'control_mode'"
                )
            control_mode = ControlMode(data["control_mode"])
            if "action_frequency_hz" not in data:
                raise ValueError(
                    "metadata.json is missing required field 'action_frequency_hz'"
                )
            action_frequency_hz = float(data["action_frequency_hz"])
            return cls(
                eval_id=eval_id,
                location=data["location"],
                robot_name=data["robot_name"],
                robot_type=robot_type,
                control_mode=control_mode,
                action_frequency_hz=action_frequency_hz,
                time=time,
                evaluator_name=data.get("evaluator_name"),
                eval_name=data.get("eval_name"),
            )


def main():
    """for testing this module"""
    import os
    import tempfile

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    # Create an instance of MetaData
    time_stamp = TimeStamp()
    eval_id = EvalID.create(time_stamp, RobotType.FRANKA)
    metadata = MetaData(
        eval_id=eval_id,
        location="Test Location",
        robot_name="Test Robot",
        robot_type=RobotType.FRANKA,
        control_mode=ControlMode.JOINT_POSITION,
        action_frequency_hz=10.0,
        time=time_stamp,
        evaluator_name="Tester",
        eval_name="unit_test_eval",
    )

    # Save the instance to the temporary file
    metadata.save(temp_file_path)

    # Load the instance from the temporary file
    loaded_metadata = MetaData.load(temp_file_path)

    # Clean up the temporary file
    os.remove(temp_file_path)


if __name__ == "__main__":
    main()
