"""Convert LIBERO RLDS dataset to LeRobot format with spatial basis action chunks.

Extends the standard conversion by computing per-timestep spatial basis
action chunks and storing them alongside the original temporal features.

The spatial basis re-parameterizes the trajectory by SE3 arc-length,
producing uniformly-spaced keyframes regardless of the robot's speed.
See ``scripts/temporal_decoupling/spatial_basis_action_chunk.py`` for details.

Stored features per frame
─────────────────────────
  *Original (unchanged)*
    image           : (256,256,3)  agent-view RGB
    wrist_image     : (256,256,3)  wrist-cam RGB
    state           : (8,)         [ee_pos(3), ee_axisangle(3), grip_qpos(2)]
    actions         : (7,)         [Δpos(3), Δaxisangle(3), grip_cmd(1)]

  *Spatial basis (added)*
    spatial_state     : (Hx8,)  spatial keyframe states  (reshape to (H,8) at train time)
    spatial_actions   : (Hx7,)  spatial action chunk     (reshape to (H,7) at train time)
    spatial_timesteps : (H,)    fractional original timestep per keyframe

Usage
-----
  uv run examples/libero/convert_libero_data_to_lerobot_change_of_basis.py \\
      --data-dir /path/to/rlds_data

  uv run examples/libero/convert_libero_data_to_lerobot_change_of_basis.py \\
      --data-dir /path/to/rlds_data --push-to-hub

Note: requires ``uv pip install tensorflow tensorflow_datasets``
Download raw data: https://huggingface.co/datasets/openvla/modified_libero_rlds
"""

from __future__ import annotations

import dataclasses
import logging
import os
os.environ["HF_LEROBOT_HOME"] = "/viscam/projects/lacwm/lerobot"
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro
import viser.transforms as vtf
import h5py
import glob
import json
# ── Import spatial basis action chunking ────────────────────────────────
from spatial_basis_action_chunk import SpatialTrajectoryIndex  # noqa: E402

logger = logging.getLogger(__name__)

REPO_NAME = "libero_aug_spatial_basis"
RAW_DATASET_NAMES = [
    "libero_10",
    "libero_goal",
    "libero_object",
    "libero_spatial",
]

# ── State / Action ↔ SE3 helpers ────────────────────────────────────────
#
# ASSUMPTION – LIBERO RLDS state layout (8-D):
#   [ee_pos(3), ee_axisangle(3), gripper_qpos(2)]
# If your RLDS variant stores orientation differently (e.g. quaternion)
# or uses joint positions, adjust the slices and conversion functions below.
#
# Action layout (7-D):
#   [Δpos(3), Δaxisangle(3), gripper_cmd(1)]

STATE_POS = slice(0, 3)
STATE_ORI = slice(3, 6)
STATE_GRIP = slice(6, 8)

ACTION_POS = slice(0, 3)
ACTION_ORI = slice(3, 6)
ACTION_GRIP = slice(6, 7)


def _axisangle_to_wxyz(aa: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle → (N, 4) wxyz quaternion."""
    xyzw = Rotation.from_rotvec(aa).as_quat()  # scipy convention: xyzw
    return xyzw[:, [3, 0, 1, 2]]


def _wxyz_to_axisangle(wxyz: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternion → (N, 3) axis-angle."""
    xyzw = wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(xyzw).as_rotvec()


def states_to_se3(states: np.ndarray) -> vtf.SE3:
    """Convert (T, 8) state array → batched vtf.SE3 of length T."""
    pos = states[:, STATE_POS]              # (T, 3)
    wxyz = _axisangle_to_wxyz(states[:, STATE_ORI])  # (T, 4)
    return vtf.SE3(wxyz_xyz=np.concatenate([wxyz, pos], axis=-1))


def actions_to_se3(actions: np.ndarray) -> vtf.SE3:
    """Convert (T, 7) action array → batched delta vtf.SE3 of length T."""
    dpos = actions[:, ACTION_POS]           # (T, 3)
    wxyz = _axisangle_to_wxyz(actions[:, ACTION_ORI])  # (T, 4)
    return vtf.SE3(wxyz_xyz=np.concatenate([wxyz, dpos], axis=-1))


def se3_to_state_rows(se3: vtf.SE3, gripper: np.ndarray) -> np.ndarray:
    """SE3 (H,) + gripper (H, G) → (H, 3+3+G) matching state layout."""
    pos = se3.translation()                              # (H, 3)
    aa = _wxyz_to_axisangle(se3.rotation().wxyz)         # (H, 3)
    return np.concatenate([pos, aa, gripper], axis=-1).astype(np.float32)


def se3_to_action_rows(se3: vtf.SE3, gripper: np.ndarray) -> np.ndarray:
    """Delta SE3 (H,) + gripper (H, G) → (H, 3+3+G) matching action layout."""
    dpos = se3.translation()                             # (H, 3)
    daa = _wxyz_to_axisangle(se3.rotation().wxyz)        # (H, 3)
    return np.concatenate([dpos, daa, gripper], axis=-1).astype(np.float32)


# ── Per-episode spatial basis computation ───────────────────────────────

def compute_spatial_basis_for_episode(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    step_size: float,
    horizon: int,
    alpha: float,
) -> list[dict[str, np.ndarray]]:
    """Compute spatial-basis features for every timestep in one episode.

    Args:
        states:  (T, 8)  proprioceptive state array.
        actions: (T, 7)  temporal action array.
        step_size: Fixed SE3 arc-length between spatial keyframes.
        horizon:   Number of spatial keyframes per chunk.
        alpha:     Rotation weight in the Lie-algebraic norm.

    Returns:
        List of T dicts each containing ``spatial_state``, ``spatial_actions``,
        and ``spatial_timesteps`` as flat float32 arrays.
    """
    T = len(states)

    # Build SE3 objects
    state_se3 = states_to_se3(states)
    action_se3 = actions_to_se3(actions)

    state_grip = states[:, STATE_GRIP]       # (T, 2)
    action_grip = actions[:, ACTION_GRIP]    # (T, 1)

    # Amortised spatial index — backbone computed once
    spatial_idx = SpatialTrajectoryIndex(
        state_se3,
        state_grip,
        alpha=alpha,
        action_se3=action_se3,
        action_gripper=action_grip,
        action_space="delta",
    )

    results: list[dict[str, np.ndarray]] = []
    for t in range(T):
        s_se3, s_grip, s_ts, a_se3, a_grip = spatial_idx.query(
            query_idx=t,
            step_size=step_size,
            horizon=horizon,
        )

        sp_state = se3_to_state_rows(s_se3, s_grip)     # (H, 8)
        sp_action = se3_to_action_rows(a_se3, a_grip)    # (H, 7)

        results.append({
            "spatial_state": sp_state.flatten(),           # (H*8,)
            "spatial_actions": sp_action.flatten(),        # (H*7,)
            "spatial_timesteps": s_ts.astype(np.float32),  # (H,)
        })

    return results


# ── CLI / main ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    data_dir: str
    """Path to the directory containing the HDF5 datasets."""

    push_to_hub: bool = False
    """Push converted dataset to the Hugging Face Hub."""

    # Spatial basis parameters
    spatial_step_size: float = 0.0075
    """Fixed SE3 arc-length distance between spatial keyframes."""

    spatial_horizon: int = 50
    """Number of spatial keyframes per chunk."""

    spatial_alpha: float = 0.01
    """Rotation weight in the SE3 Lie-algebraic norm."""


def main(args: Args) -> None:
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    H = args.spatial_horizon

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            # ── Original features ────────────────────────────────────
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            # ── Spatial basis features ───────────────────────────────
            "spatial_state": {
                "dtype": "float32",
                "shape": (H * 8,),
                "names": ["spatial_state"],
            },
            "spatial_actions": {
                "dtype": "float32",
                "shape": (H * 7,),
                "names": ["spatial_actions"],
            },
            "spatial_timesteps": {
                "dtype": "float32",
                "shape": (H,),
                "names": ["spatial_timesteps"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # ── Persist spatial basis parameters in dataset metadata ─────────
    # These end up in meta/info.json so downstream consumers know
    # exactly what conversion settings were used.
    dataset.meta.info["spatial_basis"] = {
        "step_size": args.spatial_step_size,
        "horizon": args.spatial_horizon,
        "alpha": args.spatial_alpha,
    }
    from lerobot.common.datasets.utils import write_info
    write_info(dataset.meta.info, dataset.root)

    total_episodes = 0
    for raw_dataset_name in RAW_DATASET_NAMES:
        
        h5files = glob.glob(os.path.join(args.data_dir, raw_dataset_name, "**/*.hdf5"), recursive=True)
        logger.info(f"Loading H5 dataset: {raw_dataset_name}")
        for h5file in h5files:
            raw_dataset = h5py.File(h5file, "r")

            demos = list(raw_dataset["data"].keys())
            problem_info = json.loads(raw_dataset["data"].attrs["problem_info"])
            language_instruction = problem_info["language_instruction"]
            for ep_idx, episode in enumerate(demos):
                # Collect full episode — needed for trajectory-level spatial basis
                data = raw_dataset["data"][episode]
                
                ee_states = data["obs/ee_states"]  # (T, 6)
                gripper_states = data["obs/gripper_states"] # (T, 2)
                states = np.concatenate([ee_states, gripper_states], axis=-1, dtype=np.float32) # (T, 8)
                actions_arr = data["actions"].astype(np.float32) # (T, 7)

                # Compute spatial basis for the whole episode
                spatial_data = compute_spatial_basis_for_episode(
                    states,
                    actions_arr,
                    step_size=args.spatial_step_size,
                    horizon=args.spatial_horizon,
                    alpha=args.spatial_alpha,
                )

                # Write each frame with both original and spatial features
                seq_length = states.shape[0]
                images = data["obs/agentview_rgb"]
                wrist_images = data["obs/eye_in_hand_rgb"]
                
                for step in range(seq_length):
                    
                    dataset.add_frame(
                        {
                            # Original
                            "image": images[step],
                            "wrist_image": wrist_images[step],
                            "state": states[step],
                            "actions": actions_arr[step],
                            # Spatial basis
                            "spatial_state": spatial_data[step]["spatial_state"],
                            "spatial_actions": spatial_data[step]["spatial_actions"],
                            "spatial_timesteps": spatial_data[step]["spatial_timesteps"],
                            # Task label
                            "task": language_instruction,
                        }
                    )

                dataset.save_episode()
                total_episodes += 1

                if (ep_idx + 1) % 10 == 0:
                    logger.info(f"  {raw_dataset_name}: {ep_idx + 1} episodes processed")

        logger.info(f"Finished {raw_dataset_name}")

    logger.info(
        f"Conversion complete — {total_episodes} episodes, "
        f"spatial_step_size={args.spatial_step_size}, "
        f"spatial_horizon={args.spatial_horizon}, "
        f"spatial_alpha={args.spatial_alpha}"
    )

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "h5", "spatial_basis"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
