"""Exercise safety gates through the CLI with fake motors, cameras and predictions."""

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_so101 as harness  # noqa: E402

NAMES = [f"{n}.pos" for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


class SafetyGates(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "calibration.json"
        self.path.write_text('{"steps": []}')
        self.robot = Mock()
        self.robot.action_features = dict.fromkeys(NAMES, float)
        self.robot.calibration = {
            name.removesuffix(".pos"): SimpleNamespace(id=i, range_min=0, range_max=4095)
            for i, name in enumerate(NAMES, 1)
        }
        self.robot.bus.motors = {n: SimpleNamespace(id=c.id, model="sts3215") for n, c in self.robot.calibration.items()}
        self.robot.bus.resolution.return_value = 4096
        self.robot.bus._normalize.side_effect = lambda values: {k: v / 4095 * 100 for k, v in values.items()}
        self.robot.bus.sync_read.return_value = dict.fromkeys(self.robot.calibration, 2048)
        self.robot.bus._definition.return_value = SimpleNamespace(operating_modes={"position": 0})
        self.robot.bus.read.return_value = 0
        self.robot.is_calibrated = True
        self.robot.cameras = {n: Mock(is_connected=True) for n in ("overhead", "wrist")}
        self.robot.get_observation.return_value = {
            **dict.fromkeys(NAMES, 50.0),
            **{n: np.zeros((2, 2, 3), dtype=np.uint8) for n in self.robot.cameras},
        }
        features = {"action": {"names": NAMES}, "observation.state": {"names": NAMES}}
        features.update({f"observation.images.{n}": {"shape": [2, 2, 3]} for n in self.robot.cameras})
        self.info = {"features": features, "fps": 30}
        self.policy = Mock(config=SimpleNamespace(input_features={
            "observation.state": SimpleNamespace(shape=[6]),
            **{f"observation.images.{n}": SimpleNamespace(shape=[3, 2, 2]) for n in self.robot.cameras},
        }))
        self.action = torch.full((6,), 50.0)
        self.prediction_count = 0
        self.clock = 0.0
        self.prediction_delay = 0.0

    def prediction(self, *args: object) -> torch.Tensor:
        self.prediction_count += 1
        self.clock += self.prediction_delay
        return self.action

    def invoke(self, extra: list[str], confirmation: str = "NO") -> None:
        args = ["run_so101.py", "--device", "cpu", "--task", "test", "--port", str(self.path),
                "--calibration", str(self.path), "--overhead", "0", "--wrist", "1", "--state-units", "degrees",
                "--confirm-training-semantics", "--confirm-camera-mapping", "--duration", "0.05", *extra]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", args))
            stack.enter_context(patch.object(harness, "metadata", return_value=self.info))
            stack.enter_context(patch.object(harness, "load_policy", return_value=(self.policy, Mock(), Mock())))
            stack.enter_context(patch.object(harness, "hf_hub_download", return_value=str(self.path)))
            stack.enter_context(patch.object(harness, "predict", side_effect=self.prediction))
            stack.enter_context(patch("lerobot.robots.so_follower.so_follower.SOFollower", return_value=self.robot))
            stack.enter_context(patch.object(harness.time, "monotonic", side_effect=lambda: self.clock))
            stack.enter_context(patch.object(harness.time, "sleep", side_effect=self.advance))
            stack.enter_context(patch("builtins.input", return_value=confirmation))
            harness.main()

    def advance(self, seconds: float) -> None:
        self.clock += max(seconds, 0.001)

    def assert_no_motion(self) -> None:
        self.robot.send_action.assert_not_called()
        self.robot.bus.enable_torque.assert_not_called()
        self.robot.bus.disable_torque.assert_not_called()

    def test_default_dry_run_never_commands_or_changes_torque(self) -> None:
        self.invoke([])
        self.assertGreater(self.prediction_count, 1)
        self.assert_no_motion()
        self.robot.bus.disconnect.assert_called_once_with(disable_torque=False)

    def test_execute_requires_confirmation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Motion cancelled"):
            self.invoke(["--execute", "--max-relative-target", "1"])
        self.assert_no_motion()
        self.robot.bus.disconnect.assert_called_once_with(disable_torque=False)

    def test_bad_calibration_blocks_before_inference_and_motion(self) -> None:
        self.robot.is_calibrated = False
        with self.assertRaisesRegex(RuntimeError, "calibration differs"):
            self.invoke(["--execute", "--max-relative-target", "1"], "EXECUTE")
        self.assertEqual(self.prediction_count, 0)
        self.assert_no_motion()

    def test_out_of_range_action_blocks_before_confirmation(self) -> None:
        self.action[0] = 101
        with self.assertRaisesRegex(RuntimeError, "Outside calibrated range"):
            self.invoke(["--execute", "--max-relative-target", "1"], "EXECUTE")
        self.assert_no_motion()

    def test_nonfinite_action_blocks_motion(self) -> None:
        self.action[0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Outside calibrated range"):
            self.invoke(["--execute", "--max-relative-target", "1"], "EXECUTE")
        self.assert_no_motion()

    def test_slow_prediction_never_sends_a_late_target(self) -> None:
        self.prediction_delay = 1.0
        self.invoke(["--execute", "--max-relative-target", "1"], "EXECUTE")
        # Only the explicitly approved fresh measured-pose target precedes torque enable.
        self.robot.send_action.assert_called_once_with(dict.fromkeys(NAMES, 50.0))
        self.robot.bus.enable_torque.assert_called_once()

    def test_offline_cannot_authorize_execution(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Offline data cannot authorize"):
            self.invoke(["--offline-check", "--execute"])
        self.assert_no_motion()

    def test_duration_over_five_seconds_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Duration must be"):
            self.invoke(["--duration", "6"])
        self.assert_no_motion()


if __name__ == "__main__":
    unittest.main()
