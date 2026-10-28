"""Guarded local SO-101 model testing using official LeRobot policies and processors."""

import argparse
import importlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

MODELS = json.loads(Path(__file__).with_name("models.json").read_text())
LOG = logging.getLogger("so101")


def require(condition: bool, message: str) -> None:
    """Stop rather than infer missing hardware or safety information."""
    if not condition:
        raise RuntimeError(message)


def metadata(profile: dict[str, Any]) -> dict[str, Any]:
    """Read the reference recording dataset's feature order and capture profile."""
    path = hf_hub_download(
        profile["reference_dataset"],
        "meta/info.json",
        repo_type="dataset",
        revision=profile["dataset_revision"],
    )
    return json.loads(Path(path).read_text())


def load_policy(profile: dict[str, Any], device: str) -> tuple[Any, Any, Any]:
    """Load the pinned checkpoint and its saved processor pair on the chosen device."""
    importlib.import_module(
        f"lerobot.policies.{profile['policy_type']}.configuration_{profile['policy_type']}"
    )
    config = PreTrainedConfig.from_pretrained(
        profile["repo_id"], revision=profile["revision"], cli_overrides=[f"--device={device}"]
    )
    require(config.type == profile["policy_type"], "Checkpoint policy type differs from the profile")
    require(not getattr(config, "adapt_to_pi_aloha", False), "Unexpected ALOHA conversion")
    require(not getattr(config, "use_delta_joint_actions_aloha", False), "Unexpected delta actions")
    require(not getattr(config, "use_relative_actions", False), "Unexpected relative actions")
    require(tuple(config.output_features["action"].shape) == (6,), "Expected six output positions")
    policy = (
        get_policy_class(config.type)
        .from_pretrained(profile["repo_id"], revision=profile["revision"], config=config)
        .to(device)
        .eval()
    )
    pre, post = make_pre_post_processors(
        policy.config,
        profile["repo_id"],
        pretrained_revision=profile["revision"],
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    LOG.info(
        "model=%s revision=%s device=%s dtype=%s",
        profile["repo_id"],
        profile["revision"],
        device,
        next(policy.parameters()).dtype,
    )
    policy.reset()
    pre.reset()
    post.reset()
    return policy, pre, post


def predict(policy: Any, pre: Any, post: Any, observation: dict[str, Any]) -> torch.Tensor:
    """Return one unnormalized, finite six-joint action and log latency."""
    start = time.monotonic()
    with torch.inference_mode():
        batch = pre(observation)
        require(
            batch["observation.state"].device.type == torch.device(policy.config.device).type,
            "Preprocessor is on the wrong device",
        )
        action = post(policy.select_action(batch)).detach().cpu().reshape(-1)
        if policy.config.device == "mps":
            torch.mps.synchronize()
        elif policy.config.device == "cuda":
            torch.cuda.synchronize()
    require(action.shape == (6,) and bool(torch.isfinite(action).all()), "Invalid six-dimensional action")
    LOG.info("inference_ms=%.1f predicted=%s", 1000 * (time.monotonic() - start), action.tolist())
    return action


def main() -> None:
    """Validate live inputs before allowing an explicitly confirmed short rollout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="multitask")
    parser.add_argument(
        "--list-models", action="store_true", help="Print pinned models and validation status"
    )
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--episode", type=int, default=0, help="Recorded episode for offline checks")
    parser.add_argument("--frame", type=int, default=0, help="Frame offset in that episode")
    parser.add_argument(
        "--log-file", type=Path, help="Optional log path; contains predictions and sent commands"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: never write motor registers")
    mode.add_argument(
        "--execute", action="store_true", help="Requires live validation and terminal confirmation"
    )
    parser.add_argument(
        "--offline-check", action="store_true", help="Recorded sample only; cannot authorize motion"
    )
    parser.add_argument(
        "--task", help="Live instruction; offline default is the recorded sample's exact task"
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds, maximum 5")
    parser.add_argument("--port")
    parser.add_argument("--calibration", type=Path, help="Original calibration JSON for this physical arm")
    parser.add_argument("--overhead", type=int, help="Verified OpenCV overhead index")
    parser.add_argument("--wrist", type=int, help="Verified OpenCV wrist index")
    parser.add_argument(
        "--state-units", choices=["degrees", "normalized"], help="Must match recording config"
    )
    parser.add_argument(
        "--confirm-training-semantics",
        action="store_true",
        help="Confirm original training uses absolute .pos commands in --state-units",
    )
    parser.add_argument(
        "--confirm-camera-mapping",
        action="store_true",
        help="Confirm indices, mounting, orientation and views match training",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        help="Explicit per-command LeRobot safety cap in selected joint units (gripper: percent)",
    )
    args = parser.parse_args()
    if args.list_models:
        print(json.dumps(MODELS, indent=2))
        return
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(args.log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(handler)
    profile = MODELS[args.model]
    device = args.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    require(device != "mps" or torch.backends.mps.is_available(), "Apple MPS is unavailable")
    require(device != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    require(math.isfinite(args.duration) and 0 < args.duration <= 5, "Duration must be in (0, 5] seconds")
    require(not (args.offline_check and args.execute), "Offline data cannot authorize execution")
    require(args.episode >= 0 and args.frame >= 0, "Episode/frame must be nonnegative")
    info = metadata(profile)
    names = info["features"]["action"]["names"]
    require(
        names == info["features"]["observation.state"]["names"] and len(names) == 6,
        "Dataset state/action ordering mismatch",
    )
    LOG.info("dataset_fps=%s state/action_order=%s", info["fps"], names)
    if args.offline_check:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        policy, pre, post = load_policy(profile, device)
        dataset = LeRobotDataset(
            profile["reference_dataset"],
            revision=profile["dataset_revision"],
            episodes=[args.episode],
            video_backend="pyav",
        )
        require(args.frame < len(dataset), "Frame is outside the selected episode")
        sample = dataset[args.frame]
        obs = {k: v for k, v in sample.items() if k.startswith("observation.")}
        obs["task"] = args.task or sample["task"]
        LOG.info(
            "reference_dataset=%s dataset_revision=%s episode=%d frame=%d recorded_task=%r task=%r",
            profile["reference_dataset"],
            profile["dataset_revision"],
            args.episode,
            args.frame,
            sample["task"],
            obs["task"],
        )
        LOG.info("recorded_state=%s", obs["observation.state"].tolist())
        predict(policy, pre, post, obs)
        LOG.info("OFFLINE CHECK PASSED; hardware, units, mapping and safe limits remain unverified")
        return

    require(bool(args.task and args.task.strip()), "Live inference requires an explicit --task instruction")
    require(
        all(
            x is not None for x in [args.port, args.calibration, args.overhead, args.wrist, args.state_units]
        ),
        "Supply verified --port, --calibration, --overhead, --wrist and --state-units",
    )
    require(args.calibration.is_file(), "Original calibration file is missing; no automatic calibration")
    require(Path(args.port).exists(), "Serial port is missing; connect the follower USB cable")
    require(args.overhead != args.wrist, "The two training cameras must be distinct")
    require(
        args.confirm_training_semantics and args.confirm_camera_mapping,
        "Verify original training units/absolute commands and camera mapping, then supply both confirmation flags",
    )
    if args.execute:
        require(
            args.max_relative_target is not None
            and math.isfinite(args.max_relative_target)
            and args.max_relative_target > 0,
            "Execution requires a positive --max-relative-target",
        )

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    cameras = {}
    for key in ("overhead", "wrist"):
        h, w, c = info["features"][f"observation.images.{key}"]["shape"]
        require(c == 3, "Expected RGB input")
        cameras[key] = OpenCVCameraConfig(
            index_or_path=getattr(args, key), width=w, height=h, fps=info["fps"]
        )
    robot = SOFollower(
        SOFollowerRobotConfig(
            port=args.port,
            id=args.calibration.stem,
            calibration_dir=args.calibration.parent,
            use_degrees=args.state_units == "degrees",
            cameras=cameras,
            disable_torque_on_disconnect=False,
            max_relative_target=args.max_relative_target,
        )
    )
    require(list(robot.action_features) == names, "Robot and dataset joint ordering differ")
    require(set(robot.calibration) == {n.removesuffix(".pos") for n in names}, "Incomplete calibration")
    bounds = {}
    for motor, cal in robot.calibration.items():
        max_tick = robot.bus.resolution(robot.bus.motors[motor].model) - 1
        require(
            cal.id == robot.bus.motors[motor].id and 0 <= cal.range_min < cal.range_max <= max_tick,
            f"Invalid calibration for {motor}",
        )
        # Use exactly the driver's calibrated normalization, including drive-mode inversion.
        ends = [robot.bus._normalize({cal.id: v})[cal.id] for v in (cal.range_min, cal.range_max)]
        bounds[f"{motor}.pos"] = (min(ends), max(ends))
    LOG.info(
        "port=%s calibration=%s units=%s bounds=%s cameras=%s",
        args.port,
        args.calibration,
        args.state_units,
        bounds,
        cameras,
    )
    policy, pre, post = load_policy(profile, device)
    require(
        tuple(policy.config.input_features["observation.state"].shape) == (6,), "Expected six state positions"
    )
    rename = next(
        (
            step.get("config", {}).get("rename_map", {})
            for step in json.loads(
                Path(
                    hf_hub_download(
                        profile["repo_id"], "policy_preprocessor.json", revision=profile["revision"]
                    )
                ).read_text()
            )["steps"]
            if step["registry_name"] == "rename_observations_processor"
        ),
        {},
    )
    require(
        all(
            rename.get(f"observation.images.{k}", f"observation.images.{k}") in policy.config.input_features
            for k in cameras
        ),
        "Physical views are not checkpoint inputs",
    )

    def validate(values: dict[str, float]) -> None:
        require(set(values) == set(names), "Missing/extra joint commands")
        for name, value in values.items():
            lo, hi = bounds[name]
            require(
                math.isfinite(value) and lo <= value <= hi,
                f"Outside calibrated range: {name}={value} [{lo}, {hi}]",
            )

    def observe() -> tuple[dict[str, Any], dict[str, float]]:
        raw = robot.bus.sync_read("Present_Position", normalize=False)
        for motor, value in raw.items():
            cal = robot.calibration[motor]
            require(
                cal.range_min <= value <= cal.range_max, f"Raw position outside calibration: {motor}={value}"
            )
        live = robot.get_observation()
        state = {n: float(live[n]) for n in names}
        validate(state)
        obs: dict[str, Any] = {
            "observation.state": torch.tensor([state[n] for n in names]),
            "task": args.task,
        }
        for key in cameras:
            img = live[key]
            shape = tuple(info["features"][f"observation.images.{key}"]["shape"])
            require(
                img.dtype == np.uint8 and img.shape == shape,
                f"Wrong camera shape/type: {key}: {img.shape}, {img.dtype}",
            )
            obs[f"observation.images.{key}"] = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255
        return obs, state

    try:
        # Robot.connect() calls configure(), which toggles torque. Use its existing read-only
        # bus/camera connection APIs to preserve torque, calibration and PID registers.
        robot.bus.connect()
        require(robot.is_calibrated, "Motor calibration differs from the original file; stop")
        for motor in robot.bus.motors:
            expected_mode = robot.bus._definition(motor).operating_modes["position"]
            require(
                robot.bus.read("Operating_Mode", motor) == expected_mode, f"{motor} is not in position mode"
            )
        for camera in robot.cameras.values():
            camera.connect()
        obs, state = observe()
        LOG.info("current_state=%s", state)
        action = predict(policy, pre, post, obs)
        validate(dict(zip(names, action.tolist(), strict=True)))
        LOG.info("LIVE DRY RUN PASSED: first action is finite, ordered, postprocessed and within calibration")
        deadline = time.monotonic() + args.duration
        if args.execute:
            require(
                input(
                    "Clear the workspace; keep robot power disconnect ready. Type EXECUTE for <=5s motion: "
                )
                == "EXECUTE",
                "Motion cancelled",
            )
            # Refresh the live pose after confirmation; never enable torque against a stale goal.
            _, state = observe()
            deadline = time.monotonic() + args.duration
            robot.send_action(state)
            robot.bus.enable_torque()
        policy.reset()
        pre.reset()
        post.reset()
        executed = 0
        while time.monotonic() < deadline:
            start = time.monotonic()
            obs, _ = observe()
            action = predict(policy, pre, post, obs)
            targets = dict(zip(names, action.tolist(), strict=True))
            validate(targets)
            # Inference may cross the deadline. Never send a late command or catch up in bursts.
            if time.monotonic() >= deadline:
                break
            if args.execute:
                applied = robot.send_action(targets)
                executed += 1
                LOG.info("executed=%d applied=%s", executed, applied)
            time.sleep(max(0, min(1 / info["fps"] - (time.monotonic() - start), deadline - time.monotonic())))
        LOG.info("Finished: executed=%d; torque retained; last commanded target may still settle", executed)
    finally:
        # Release each resource even after partial connection. Do not issue shutdown motion.
        for camera in robot.cameras.values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception:
                    LOG.exception("Camera cleanup failed")
        robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning(
            "Interrupted; no further commands. Torque retained. Cut motor power for an emergency stop."
        )
        raise SystemExit(130) from None
    except Exception:
        LOG.exception("STOPPED")
        raise SystemExit(1) from None
