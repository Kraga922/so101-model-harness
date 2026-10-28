# SO-101 model test harness

Run EIG's published SmolVLA and Pi0.5 checkpoints through LeRobot's official policy, processor, camera and SO-101 motor APIs. Start with recorded observations, then a live dry-run, then an explicitly confirmed five-second physical test.

**Physical rollout has not been validated on hardware.** A successful recorded-data check verifies the inference pipeline, not task success or safe physical operation. Do not leave a running arm unattended.

## Models

| CLI name | Checkpoint | Task |
|---|---|---|
| `multitask` | [TBD-smolvla-so101-multitask-208](https://huggingface.co/ufeig/TBD-smolvla-so101-multitask-208) | Zone movement, two-cap stacking, red/black-cap bowl placement |
| `white-bowl` | [smolvla-white-caps-bowl](https://huggingface.co/ufeig/smolvla-white-caps-bowl) | White-cap bowl placement |
| `zone` | [cad-smolvla-so101-intozone](https://huggingface.co/ufeig/cad-smolvla-so101-intozone) | Red/black/white cap to zone |
| `pi05` | [hyperion-pi0.5](https://huggingface.co/ufeig/hyperion-pi0.5) | White-cap bowl placement; experimental loader path |

`models.json` pins checkpoint and reference dataset revisions and records validation status. Model weights stay on Hugging Face and download into its normal cache; this repository contains no weights, tokens, recordings or robot calibration. The complete dataset audit and exact task strings are in [MODEL_TEST_PLAN.md](MODEL_TEST_PLAN.md). Two merged training datasets are inaccessible: their reference recordings must not be described as certified held-out data.

## Install

Requirements: Python 3.12 or 3.13, Git, [uv](https://docs.astral.sh/uv/), internet access for dependency/model downloads, and sufficient disk space. MPS is available on compatible Apple Silicon Macs; CUDA and CPU are also selectable. CPU inference may be too slow for physical use. Pi0.5 memory requirements and device compatibility have not been validated by this project.

Download/clone this repository, enter its directory, then:

```bash
uv sync
# For Pi0.5's additional dependencies:
uv sync --extra pi
# If a Hugging Face repository requires authentication:
uv run hf auth login
uv run python run_so101.py --list-models
```

LeRobot is pinned to Git commit `9f45b7b37cc4b33f8ecdb503aa4d209370bd528d`, rather than relying on a potentially different PyPI release with the same version. The original code lives at [huggingface/lerobot](https://github.com/huggingface/lerobot). Dependencies are managed by this directory's `pyproject.toml`; no edits to LeRobot training code are required.

A clean standalone installation is not yet verified. The dependency uses GitHub's archive of the pinned commit, avoiding a full Git fetch (the initial fetch failed with an HTTP/2 disconnect). Development tests used an existing editable installation at the same LeRobot commit. Only the multitask weights have completed local inference validation; the other model profiles are experimental, and large downloads may take time. All four saved configs parse successfully with the pinned LeRobot code.

## 1. Test without a robot

```bash
uv run python run_so101.py --model multitask --offline-check --device auto
uv run python run_so101.py --model white-bowl --offline-check
uv run python run_so101.py --model zone --offline-check
uv run --extra pi python run_so101.py --model pi05 --offline-check
```

Each run loads its pinned reference dataset, uses the recorded sample's **exact task instruction** by default, and runs saved preprocessing → policy inference → saved action postprocessing. It prints the recorded state, model/data revisions, instruction, latency and six predicted positions. It does not open the robot serial port. `--episode N --frame N` selects another moment. `--task '...'` deliberately overrides the recorded instruction; that becomes a separate prompt experiment. No synthetic image is presented as a real camera.

The white-bowl reference dataset is a related recording with a matching task string, not its inaccessible merged training dataset. A reference example is not automatically a held-out evaluation.

Optional logs:

```bash
uv run python run_so101.py --model multitask --offline-check --log-file logs/multitask.log
```

## 2. Prepare your own hardware

Use an assembled, powered and previously calibrated SO-101 follower. Follow [LeRobot's SO-101 setup guide](https://huggingface.co/docs/lerobot/main/en/so101) for initial motor setup and calibration; the harness will not perform these operations. Do not borrow somebody else's calibration JSON. Preserve the original recording unit convention and equivalent joint zero references when preparing a different arm.

1. Identify the follower serial port and overhead/wrist cameras:

   ```bash
   uv run python -c 'import serial.tools.list_ports; print([(p.device,p.description) for p in serial.tools.list_ports.comports()])'
   uv run lerobot-find-cameras opencv
   ```

2. Inspect camera frames. Match the training demonstrations' mounting, orientation, workspace, cap colors/types and targets. On macOS, grant Camera permission to the terminal/app running the process. Indices are machine-specific and may change after reconnection.
3. Obtain the original calibration JSON for **your arm**, confirm its calibration matches the motors, and confirm whether recording used `use_degrees=true` (`--state-units degrees`) or false (`normalized`). The five arm joints use degrees or calibrated [-100,100]; the gripper uses calibrated [0,100] in either case. Metadata alone does not prove these units.
4. Verify that training used absolute `.pos` commands with no custom coordinate/action conversion. Do not assert the confirmation flags below unless these facts and the camera mapping are established.

All reference recordings have overhead/wrist RGB views at 640×480 and 30 Hz. The white-bowl/zone saved processors rename them to camera1/camera2. The zone checkpoint also lists camera3, but its dataset has only two real views: the official SmolVLA implementation supplies a masked empty slot. Do not invent an unmasked third view. The harness preserves official resizing and masking behavior.

## 3. Live dry-run (default)

Set these shell variables to verified local values before running. There are deliberately no guessed serial ports, camera indices, calibration paths or training units:

```bash
: "${SO101_PORT:?Set your verified follower port}"
: "${SO101_CALIBRATION:?Set your original calibration JSON path}"
: "${SO101_OVERHEAD:?Set your verified overhead camera index}"
: "${SO101_WRIST:?Set your verified wrist camera index}"
: "${SO101_UNITS:?Set degrees or normalized to match recording}"
: "${SO101_TASK:?Set the exact intended task instruction}"
common=(--model multitask --port "$SO101_PORT" --calibration "$SO101_CALIBRATION"
        --overhead "$SO101_OVERHEAD" --wrist "$SO101_WRIST"
        --state-units "$SO101_UNITS" --task "$SO101_TASK"
        --confirm-training-semantics --confirm-camera-mapping)
uv run python run_so101.py "${common[@]}" --dry-run --log-file logs/dry-run.log
```

These array commands work in Bash or Zsh. Select another model in `common` as needed; for Pi0.5 use `uv run --extra pi ...`.

Dry-run opens LeRobot's bus/camera connections, verifies calibration/motor position mode, reads live images and positions, and checks postprocessed targets against calibrated joint bounds. It never writes goal registers, configures motors, changes torque or writes calibration. It logs current state and predictions for inspection. Calibration bounds do not detect collisions or establish that a trajectory is appropriate for your workspace.

## 4. Explicitly supervised physical startup test

After live dry-run passes and you approve the setup, choose a conservative positive per-command movement cap in the selected joint units (gripper percent). LeRobot's `max_relative_target` limiter applies to each command; this is not a velocity/acceleration guarantee.

```bash
: "${SO101_MAX_RELATIVE_TARGET:?Choose a conservative positive per-command cap}"
uv run python run_so101.py "${common[@]}" --execute --duration 5 \
  --max-relative-target "$SO101_MAX_RELATIVE_TARGET" --log-file logs/physical.log
```

The program repeats live checks and waits for you to type `EXECUTE` immediately before the first motor write. It sets the current measured pose as the initial target and enables torque, then sends validated predictions using `SOFollower.send_action`. It does not recalibrate or toggle torque off during setup. The maximum command window is five seconds, including initial target/torque setup; delayed inference never triggers a burst of catch-up commands.

**Emergency stop: keep the motor power disconnect within reach; cut motor power immediately if needed, supporting the arm if it could fall.** USB disconnection alone does not stop a powered servo. Ctrl+C/exceptions stop further commands and close connections; torque and the last target remain, so the arm can still settle toward that target. No automatic return pose or torque-off shutdown occurs.

Five seconds is a startup check. Full demonstration tasks often take longer. This first release deliberately refuses longer physical trials; it does not yet provide a validated full-duration task-evaluation mode. See the test plan for success criteria and demonstration durations.

## Timing and interpretation

The harness preserves the checkpoint's 50/50 chunk/action settings. It reads observations each step; a queued policy consumes new observations when it refills its action chunk. The loop targets the recording frequency (30 Hz), but synchronous inference adds refill pauses. A successful software check does not demonstrate the robot can complete a task. Log each prompt, layout, completion and intervention separately when collecting physical results.

## Development checks

```bash
uv run python -m unittest discover -s tests -v
uv run --extra dev ruff check .
```

Safety tests use fake hardware. They check dry-run write suppression, confirmation, calibration failure, invalid targets, deadlines and conflicting CLI modes; they do not certify physical safety.

## License

Apache-2.0; see [LICENSE](LICENSE). LeRobot retains its upstream copyright and license. Models and datasets remain subject to the terms of their respective Hugging Face repositories.
