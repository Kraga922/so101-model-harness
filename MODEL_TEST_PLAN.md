# Model and dataset audit for physical SO-101 tests

Inspected the authenticated Hugging Face API on 2026-10-27: all four model repositories returned for `ufeig`, their saved training configs and processor configs, both available model cards, all 30 dataset repositories returned for `ufeig`, plus the external dataset named by the zone checkpoint. This audit covers metadata and training provenance. Current harness validation status is recorded separately in `models.json`; no physical rollout has been run.

## What each model was trained for

| Model | Training dataset recorded in checkpoint | Intended test | Evidence / qualification |
|---|---|---|---|
| [TBD-smolvla-so101-multitask-208](https://huggingface.co/ufeig/TBD-smolvla-so101-multitask-208) | `ufeig/so101-multitask-208-local_train`, loaded from a training-machine local directory | Move caps into a zone; stack TWO caps; put red caps into a bowl; put black caps into a bowl | Model card: 208 total episodes, split 166 train / 21 validation / 21 test; four families, 11 instructions. The aggregate dataset is not accessible with current authentication, so exact source membership, episode split and all 11 verbatim labels cannot be certified. |
| [smolvla-white-caps-bowl](https://huggingface.co/ufeig/smolvla-white-caps-bowl) | `ufeig/bottlecaps-white-bowl`, loaded from a local directory | Pick up a WHITE cap and release it into the bowl | Aggregate dataset is not accessible. Card lists 61 episodes and three exact task labels; its prose also says 60 episodes, a discrepancy. Training config uses an 85/15 split. |
| [cad-smolvla-so101-intozone](https://huggingface.co/ufeig/cad-smolvla-so101-intozone) | [jsoon33/eig-so101-intozone-set](https://huggingface.co/datasets/jsoon33/eig-so101-intozone-set) | Move a WHITE, BLACK or RED cap to the zone, one instruction at a time | Dataset accessible: 60 episodes, 20 per instruction, 30 Hz, overhead/wrist images. |
| [hyperion-pi0.5](https://huggingface.co/ufeig/hyperion-pi0.5) | [ufeig/bottlecaps-white-robocops-2](https://huggingface.co/datasets/ufeig/bottlecaps-white-robocops-2) | Pick up a WHITE cap and release it into the bowl | Dataset accessible: 11 episodes, one instruction, 30 Hz. Saved training config requests only 300 updates; without a checkpoint manifest that is configured training length, not independent proof of the weight file's completed update count. |

Sources: each model's `train_config.json`; multitask [model card](https://huggingface.co/ufeig/TBD-smolvla-so101-multitask-208/blob/main/README.md) and [checkpoint manifest](https://huggingface.co/ufeig/TBD-smolvla-so101-multitask-208/blob/main/checkpoint_manifest.json); white-cap [model card](https://huggingface.co/ufeig/smolvla-white-caps-bowl/blob/main/README.md); dataset `meta/info.json`, `meta/tasks.parquet` and episode metadata. The multitask manifest identifies the released checkpoint as update 35,000, selected by validation; it reports no physical success rate.

**Do not treat every dataset in the account as training data for every model.** In particular, white-cap bowl placement and three-cap stacking are not task families claimed for the multitask checkpoint. Its 208 episodes describe the complete collection before splitting, not 208 training episodes.

## Exact instructions to use

Zone model: these three strings are verified directly from its training dataset:

```text
Move the white bottlecap to the zone
Move the black bottlecap to the zone
Move the red bottlecap to the zone
```

Pi0.5: this is its training dataset's sole task string:

```text
Pick up the white bottle cap and place it in the bowl
```

White-cap SmolVLA: its model card lists three variants. Keep them as distinct prompt conditions when evaluating; start with the clean `bottlecap` spelling. The first line contains the original label typo:

```text
Pick up the white botblack cap and place it in the bowl
Pick up the white bottle cap and place it in the bowl
Pick up the white bottlecap and place it in the bowl
```

Multitask: the model card certifies task families and 11 labels, but does not publish the complete label table. The accessible source recordings contain these 11 matching candidate instructions. **This is a source-derived candidate test list, not a verified reconstruction of the inaccessible aggregate.** Recover the aggregate's `meta/tasks.parquet` and source/split manifest from the training machine to confirm the exact labels before calling this a full training-task evaluation.

```text
Move the red bottlecap to the zone
Move the black bottlecap to the zone
Move the white bottlecap to the zone
Place the black bottlecap on the white bottlecap
Place the red bottlecap on the black bottlecap
Place the white bottlecap on the red bottlecap
Place the white bottlecap on the black bottlecap
Place the black bottlecap on the red bottlecap
Place the red bottlecap on the white bottlecap
Pick up the red bottlecap and place it in the bowl
Pick up the black bottlecap and place it in the bowl
```

The account also contains a black-cap dataset with the conflicting label `Pick up the red botblack cap and place it in the bowl`. Its inclusion in the multitask aggregate is not established. Do not silently substitute that label or claim that the clean black-cap prompt is confirmed for the aggregate.

## Inputs differ between models

All accessible recording datasets use six joint-position features in this order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (each suffixed `.pos`). All inspected populated datasets record at 30 Hz. Recording metadata does not establish the original degree/normalized-unit setting or arm calibration.

| Model | Saved camera inputs / mapping | Deployment consequence |
|---|---|---|
| Multitask SmolVLA | `overhead`, `wrist`, both 640×480 RGB | The harness selects this checkpoint with `--model multitask`. It has passed a recorded-observation MPS check; physical checks remain pending. |
| White-cap SmolVLA | Physical `overhead` → `camera1`, `wrist` → `camera2`; 640×480 RGB; one masked empty-camera slot | Preserve the checkpoint's saved rename processor and missing-camera handling. Do not feed a fabricated real third view. Not yet loaded/tested locally. |
| Zone SmolVLA | Saved schema lists `camera1`, `camera2`, `camera3` at 256×256 and an empty slot; saved rename maps overhead/wrist to camera1/camera2. Actual training dataset has only overhead/wrist at 640×480 | Schema and dataset differ. Current official SmolVLA image preparation accepts the two real mapped views and appends one masked image. That path was checked without weights; full checkpoint inference with actual training observations remains to be tested. Do not infer a third physical camera is required or invent one. |
| Pi0.5 | `overhead`, `wrist`, 640×480 RGB | Uses Pi0.5, QUANTILES normalization and its own tokenizer/state preparation. It cannot be run through the SmolVLA-specific loader. MPS loading/latency/memory have not been tested. |

All four configs specify 50-action chunks and 50 executed actions per chunk. SmolVLA models use MEAN_STD state/action normalization. Pi0.5 uses QUANTILES, with relative actions disabled. Each model must use its own saved processors and statistics.

## How to get a meaningful physical result

1. Open several demonstration videos for the intended task using the dataset links below. Match camera mounting/orientation, cap type and color, cap orientation, target bowl/zone, background and starting arm pose. Object positions and motions must come from the videos, not just the task wording. Viewer links are provided; video scene geometry was not visually audited in this metadata pass.
2. Complete camera mapping, original calibration and training-unit verification. Run live dry-run checks with the selected checkpoint's own processors. Do not replay demonstrations to check setup: replay sends movement commands.
3. Keep the first physical attempt to the existing five-second supervised safety test. **Five seconds is a startup test, not a fair full-task success test.** The episode-duration table below records actual demonstration lengths. After safe operation is established, completion trials need a deliberately configured longer duration based on the matching demonstration; the current runner intentionally refuses durations over five seconds.
4. Run one instruction per trial. Use the same initial layouts and time limit across models when testing an overlapping task. Compare zone SmolVLA vs multitask on each zone color; white SmolVLA vs Pi0.5 on white-cap bowl placement. Multitask has no documented white-cap bowl family, so do not score it as failing that training task.
5. Suggested initial evaluation: ten independent resets per instruction. Record model revision, exact prompt, layout, duration, success, wrong-object selection, timeout and operator intervention. Every intervention makes that trial an assisted attempt; report it separately from autonomous success. Report each instruction separately rather than averaging away failures on particular colors.
6. Define success before running: intended cap released in the bowl; intended cap inside the demonstrated zone; or requested cap released on the specified lower cap and still balanced after the gripper withdraws. Record partial progress separately. A finite predicted action or low offline loss is not task completion.

Suggested log columns:

```text
model_revision,task,trial,layout_id,time_limit_s,completed,wrong_object,timeout,intervention,notes
```

The exact saved training/validation/test membership for the local merged datasets is missing. Fresh physical trials can measure current physical success, but cannot honestly be labeled held-out training-split tests until that provenance is recovered. Curation and merged datasets can contain the same source episodes; episode counts below must not be added as unique demonstrations.

## Complete discovered dataset inventory

The following inventory preserves exact task labels, including typos. Counts/durations come from metadata; dates and repository names alone do not prove model membership. Two training dataset references above returned 404 with current authentication (missing or inaccessible, not proof they never existed). One listed ufeig repository contains only `.gitattributes`.

| Dataset | Episodes | Exact task label(s) | Episode seconds: median (min–max) |
|---|---:|---|---|
| [jsoon33/eig-so101-intozone-set](https://huggingface.co/datasets/jsoon33/eig-so101-intozone-set) | 60 | Move the white bottlecap to the zone<br>Move the black bottlecap to the zone<br>Move the red bottlecap to the zone | 9.4 (6.4–14.87) |
| [ufeig/bottlecaps-black-robocops](https://huggingface.co/datasets/ufeig/bottlecaps-black-robocops) | 20 | Pick up the red botblack cap and place it in the bowl | 8.58 (2.37–25.33) |
| [ufeig/bottlecaps-black-test-001_20261016_162948](https://huggingface.co/datasets/ufeig/bottlecaps-black-test-001_20261016_162948) | 20 | Pick up the black bottlecap and place it in the bowl | 9.95 (7.13–15.57) |
| [ufeig/bottlecaps-black-test-001_20261016_163747](https://huggingface.co/datasets/ufeig/bottlecaps-black-test-001_20261016_163747) | 5 | Pick up the black bottlecap and place it in the bowl | 9.37 (8.83–9.67) |
| [ufeig/bottlecaps-red-robocops_20261020_123956](https://huggingface.co/datasets/ufeig/bottlecaps-red-robocops_20261020_123956) | 20 | Pick up the red bottlecap and place it in the bowl | 8.02 (6.77–20.77) |
| [ufeig/bottlecaps-red-test-001_20261016_153140](https://huggingface.co/datasets/ufeig/bottlecaps-red-test-001_20261016_153140) | 2 | Pick up the red bottlecap and place it in the bowl | 16.38 (16.17–16.6) |
| [ufeig/bottlecaps-red-test-001_20261016_155411](https://huggingface.co/datasets/ufeig/bottlecaps-red-test-001_20261016_155411) | 6 | Pick up the red bottlecap and place it in the bowl | 14.07 (0.93–27.93) |
| [ufeig/bottlecaps-red-test-001_20261016_155818](https://huggingface.co/datasets/ufeig/bottlecaps-red-test-001_20261016_155818) | 19 | Pick up the red bottlecap and place it in the bowl | 13.37 (1.7–23.77) |
| [ufeig/bottlecaps-red-test-001_20261016_160736](https://huggingface.co/datasets/ufeig/bottlecaps-red-test-001_20261016_160736) | 23 | Pick up the red bottlecap and place it in the bowl | 8.63 (0.7–29.93) |
| [ufeig/bottlecaps-red-test-001_20261016_161347](https://huggingface.co/datasets/ufeig/bottlecaps-red-test-001_20261016_161347) | 5 | Pick up the red bottlecap and place it in the bowl | 11.5 (2.43–12.57) |
| [ufeig/bottlecaps-red-test-001_20261016_171600](https://huggingface.co/datasets/ufeig/bottlecaps-red-test-001_20261016_171600) | 10 | Pick up the red bottlecap and place it in the bowl | 9.12 (7.9–16.7) |
| [ufeig/bottlecaps-white-robocops](https://huggingface.co/datasets/ufeig/bottlecaps-white-robocops) | 20 | Pick up the white botblack cap and place it in the bowl | 11.13 (5.13–29.9) |
| [ufeig/bottlecaps-white-robocops-2](https://huggingface.co/datasets/ufeig/bottlecaps-white-robocops-2) | 11 | Pick up the white bottle cap and place it in the bowl | 18.4 (11.13–29.9) |
| [ufeig/bottlecaps-white-robocops_20261020_130057](https://huggingface.co/datasets/ufeig/bottlecaps-white-robocops_20261020_130057) | — | No data/metadata uploaded | — |
| [ufeig/bottlecaps-white-test-001_20261016_164211](https://huggingface.co/datasets/ufeig/bottlecaps-white-test-001_20261016_164211) | 20 | Pick up the white bottlecap and place it in the bowl | 12.82 (10.17–22.13) |
| [ufeig/bottlecaps-white-test-001_20261016_165757](https://huggingface.co/datasets/ufeig/bottlecaps-white-test-001_20261016_165757) | 10 | Pick up the white bottlecap and place it in the bowl | 11.43 (8.63–29.8) |
| [ufeig/move-to-zone-cad_20261023_153231](https://huggingface.co/datasets/ufeig/move-to-zone-cad_20261023_153231) | 20 | Move the red bottlecap to the zone | 8.2 (6.53–13.27) |
| [ufeig/move-to-zone-cad_20261023_154238](https://huggingface.co/datasets/ufeig/move-to-zone-cad_20261023_154238) | 20 | Move the black bottlecap to the zone | 10.32 (8.93–14.87) |
| [ufeig/move-to-zone-cad_20261023_155239](https://huggingface.co/datasets/ufeig/move-to-zone-cad_20261023_155239) | 20 | Move the white bottlecap to the zone | 8.83 (6.4–14.6) |
| [ufeig/red-cap-clean-10-17](https://huggingface.co/datasets/ufeig/red-cap-clean-10-17) | 35 | Pick up the red bottlecap and place it in the bowl | 9.83 (6.9–23.77) |
| [ufeig/stack-2-bottlecaps-hyperion_20261022_181332](https://huggingface.co/datasets/ufeig/stack-2-bottlecaps-hyperion_20261022_181332) | 10 | Place the black bottlecap on the white bottlecap | 9.4 (8.27–11.53) |
| [ufeig/stack-2-bottlecaps-hyperion_20261022_181730](https://huggingface.co/datasets/ufeig/stack-2-bottlecaps-hyperion_20261022_181730) | 10 | Place the red bottlecap on the black bottlecap | 12.95 (10.93–23.47) |
| [ufeig/stack-2-bottlecaps-hyperion_20261022_182304](https://huggingface.co/datasets/ufeig/stack-2-bottlecaps-hyperion_20261022_182304) | 10 | Place the white bottlecap on the red bottlecap | 8.43 (7.03–13.9) |
| [ufeig/stack-2-bottlecaps-hyperion_20261022_182607](https://huggingface.co/datasets/ufeig/stack-2-bottlecaps-hyperion_20261022_182607) | 1 | Place the white bottlecap on the black bottlecap | 18.3 (18.3–18.3) |
| [ufeig/stack-2-bottlecaps-hyperion_20261022_182654](https://huggingface.co/datasets/ufeig/stack-2-bottlecaps-hyperion_20261022_182654) | 10 | Place the white bottlecap on the black bottlecap | 13.18 (10.93–21.2) |
| [ufeig/stack-2-bottlecaps-hyperion_20261022_183059](https://huggingface.co/datasets/ufeig/stack-2-bottlecaps-hyperion_20261022_183059) | 10 | Place the black bottlecap on the red bottlecap | 9.35 (7.57–15.6) |
| [ufeig/stack-2-bottlecaps-hyperion_20261022_183430](https://huggingface.co/datasets/ufeig/stack-2-bottlecaps-hyperion_20261022_183430) | 10 | Place the white bottlecap on the red bottlecap | 10.92 (8.87–17.67) |
| [ufeig/stack-2-bottlecaps-hyperion_20261022_183855](https://huggingface.co/datasets/ufeig/stack-2-bottlecaps-hyperion_20261022_183855) | 10 | Place the red bottlecap on the white bottlecap | 10.28 (8.53–11.6) |
| [ufeig/stack-2-bottlecaps-hyperion_20261022_184400](https://huggingface.co/datasets/ufeig/stack-2-bottlecaps-hyperion_20261022_184400) | 2 | Place the red bottlecap on the white bottlecap | 8.75 (8.13–9.37) |
| [ufeig/stack-3-bottlecaps-hyperion_20261022_184621](https://huggingface.co/datasets/ufeig/stack-3-bottlecaps-hyperion_20261022_184621) | 10 | Place the black bottlecap on the red bottlecap and then the white bottlecap on the black bottlecap | 19.88 (15.8–29.67) |
| [ufeig/stack-3-bottlecaps-hyperion_20261022_185240](https://huggingface.co/datasets/ufeig/stack-3-bottlecaps-hyperion_20261022_185240) | 3 | Place the black bottlecap on the red bottlecap and then the white bottlecap on the black bottlecap | 17.33 (14.2–17.53) |

### Inspect the recorded scenes

Open a dataset in the [LeRobot dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) and enter its repository ID. Representative starting points:

- White-cap bowl (Pi0.5 exact training dataset): [view demonstrations](https://huggingface.co/spaces/lerobot/visualize_dataset?path=ufeig/bottlecaps-white-robocops-2).
- Zone tasks (zone model exact training dataset): [view demonstrations](https://huggingface.co/spaces/lerobot/visualize_dataset?path=jsoon33/eig-so101-intozone-set).
- Two-cap stacking source example; aggregate membership unverified: [view demonstrations](https://huggingface.co/spaces/lerobot/visualize_dataset?path=ufeig/stack-2-bottlecaps-hyperion_20261022_181332).
- Red-cap bowl source example; aggregate membership unverified: [view demonstrations](https://huggingface.co/spaces/lerobot/visualize_dataset?path=ufeig/red-cap-clean-10-17).
- Black-cap bowl source example; aggregate membership unverified: [view demonstrations](https://huggingface.co/spaces/lerobot/visualize_dataset?path=ufeig/bottlecaps-black-test-001_20261016_162948).
