"""Spatial basis action chunking for SE3 trajectories.

Converts a temporal trajectory into spatially-uniform keyframes by
re-parameterizing along Lie-algebra arc-length.

Two APIs are provided:
  - ``spatial_action_chunk_SE3``: stateless drop-in, recomputes everything each call.
  - ``SpatialTrajectoryIndex``: pre-computes the trajectory backbone once,
    then answers many ``query()`` calls cheaply (ideal for interactive scrubbing).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import viser.transforms as vtf

from typing import Literal

# ── Stateless drop-in, recomputes traj deltas each call ─────────────────
def spatial_action_chunk_SE3(
    state_se3: vtf.SE3,
    state_gripper: np.ndarray,
    query_idx: int,
    step_size: float,
    horizon: int,
    alpha: float = 0.01,
    action_se3: vtf.SE3 | None = None,
    action_gripper: np.ndarray | None = None,
    action_space: Literal["absolute", "delta"] = "delta",
) -> Tuple[vtf.SE3, np.ndarray, np.ndarray, vtf.SE3 | None, np.ndarray | None]:
    """Change trajectory from temporal to spatial basis (vectorised).

    Args:
        state_se3: The full SE3 trajectory (temporal basis), shape (T,).
        state_gripper: Gripper state array, shape (T,) or (T, D).
            Linearly interpolated at the same fractional positions as the SE3 poses.
        query_idx: The current time-index (e.g., from a 30Hz video frame).
        step_size: The fixed Lie-norm distance between spatial keyframes.
        horizon: How many spatial keyframes to include in the chunk.
        alpha: Weight for rotation in the norm calculation.

    Returns:
        A tuple of:
          - An SE3 object of length *horizon*, the spatial look-ahead.
          - A float array of length *horizon*, the fractional original timestep
            for each keyframe.
          - Interpolated gripper values of length *horizon*, same shape as
            state_gripper along non-time dimensions.
    """
    T = len(state_se3.parameters())
    assert horizon > 0, "Horizon must be positive"
    assert step_size > 0, "Step size must be positive"
    assert alpha >= 0, "Alpha must be non-negative"
    assert query_idx >= 0, "Query index must be non-negative"
    assert query_idx < T, "Query index must be less than the number of original poses"
    assert len(state_gripper) == T, (
        f"Gripper length ({len(state_gripper)}) must match poses ({T})"
    )

    mats = state_se3.as_matrix()  # (T, 4, 4)
    inv_mats = state_se3.inverse().as_matrix()  # (T, 4, 4)

    # Vectorised relative transforms & log maps
    rel_mats = np.einsum("nij,njk->nik", inv_mats[:-1], mats[1:])  # (T-1, 4, 4)
    deltas = vtf.SE3.from_matrix(rel_mats).log()  # (T-1, 6)

    # Cumulative arc-length
    trans_norms = np.linalg.norm(deltas[:, :3], axis=-1)
    rot_norms = np.linalg.norm(deltas[:, 3:], axis=-1)
    step_dists = np.sqrt(trans_norms**2 + alpha * rot_norms**2)
    cumulative_dist = np.concatenate(([0.0], np.cumsum(step_dists)))
    total_dist = cumulative_dist[-1]

    # Target distances
    current_s = cumulative_dist[query_idx]
    target_s = current_s + np.arange(1, horizon + 1) * step_size
    s_clamped = np.minimum(target_s, total_dist)

    # Vectorised segment lookup + LERP
    indices = np.searchsorted(cumulative_dist, s_clamped) - 1
    indices = np.clip(indices, 0, len(deltas) - 1)

    seg_starts = cumulative_dist[indices]
    seg_ends = cumulative_dist[indices + 1]
    denom = seg_ends - seg_starts
    lerp_t = np.where(denom < 1e-15, 0.0, (s_clamped - seg_starts) / denom)

    # Vectorised twist interpolation (ScLERP)
    interp_twists = lerp_t[:, None] * deltas[indices]  # (H, 6)
    delta_poses = vtf.SE3.exp(interp_twists)  # batched

    base_mats = mats[indices]  # (H, 4, 4)
    res_mats = np.einsum("nij,njk->nik", base_mats, delta_poses.as_matrix())
    interp_state_se3 = vtf.SE3.from_matrix(res_mats)
    interp_timesteps = indices.astype(np.float64) + lerp_t

    # if trajectory end is reached, keep interp timesteps counting up instead of flatlining
    at_goal_mask = target_s > total_dist
    if np.any(at_goal_mask):
        # Find the index of the first point that hits the goal
        first_goal_idx = np.argmax(at_goal_mask)
        # Calculate how many steps we are past the goal
        steps_past_goal = np.arange(len(interp_timesteps)) - first_goal_idx
        # Apply virtual increment only to points past the first goal point
        # This turns [96, 96, 96] into [96, 97, 98]
        interp_timesteps[at_goal_mask] = interp_timesteps[first_goal_idx] + steps_past_goal[at_goal_mask]


    # Linearly interpolate gripper at the same fractional positions
    gripper = state_gripper.astype(np.float64)
    next_indices = np.minimum(indices + 1, T - 1)
    interp_state_gripper = (
        (1.0 - lerp_t)[..., None] * gripper[indices]
        + lerp_t[..., None] * gripper[next_indices]
    )
    # Squeeze trailing dim if original was 1-D
    if state_gripper.ndim == 1:
        interp_state_gripper = interp_state_gripper.squeeze(-1)

    # ── Action Interpolation Logic ────────────────────────────────────
    if action_se3 is not None:
        assert len(action_se3.parameters()) == T, (
            f"action_se3 length ({len(action_se3.parameters())}) must match state ({T})"
        )
        assert action_gripper is not None, "action_gripper required when action_se3 is provided"
        assert len(action_gripper) == T, (
            f"action_gripper length ({len(action_gripper)}) must match state ({T})"
        )

        # 1. Unroll to Absolute Target Space
        # If delta: G_t = P_t * Delta_t (The pose the expert was aiming for)
        # If absolute: G_t = Action_t (The target pose directly)
        if action_space == "delta":
            expert_targets = vtf.SE3.from_matrix(
                np.einsum("nij,njk->nik", state_se3.as_matrix(), action_se3.as_matrix())
            )
        else:
            expert_targets = action_se3

        # 2. Compute Relative Twists between expert targets for ScLERP
        target_mats = expert_targets.as_matrix()
        target_inv_mats = expert_targets.inverse().as_matrix()
        
        target_deltas = vtf.SE3.from_matrix(
            np.einsum("nij,njk->nik", target_inv_mats[:-1], target_mats[1:])
        ).log()  # Shape (T-1, 6)

        # 3. Interpolate Absolute Targets
        # Apply fractional twist to the starting target pose of the segment
        interp_target_twists = lerp_t[:, None] * target_deltas[indices]
        delta_target_poses = vtf.SE3.exp(interp_target_twists)
        
        res_target_mats = np.einsum("nij,njk->nik", target_mats[indices], delta_target_poses.as_matrix())
        interp_absolute_targets = vtf.SE3.from_matrix(res_target_mats)

        # 4. Project back to Delta if necessary
        if action_space == "delta":
            # Action_new = Interp_State^-1 * Interp_Target
            # This gives the relative transform from our new spatial pose to the expert's goal
            interp_action_se3 = vtf.SE3.from_matrix(
                np.einsum("nij,njk->nik", interp_state_se3.inverse().as_matrix(), interp_absolute_targets.as_matrix())
            )
        else:
            interp_action_se3 = interp_absolute_targets

        # 5. Linear interpolation for Action Gripper
        act_gripper = action_gripper.astype(np.float64)
        interp_action_gripper = (
            (1.0 - lerp_t)[..., None] * act_gripper[indices]
            + lerp_t[..., None] * act_gripper[next_indices]
        )
        # Squeeze trailing dim if original was 1-D
        if action_gripper.ndim == 1:
            interp_action_gripper = interp_action_gripper.squeeze(-1)

        # 6. Handling the Goal Extrapolation
        if np.any(at_goal_mask):
            goal_indices = np.where(at_goal_mask)[0]
            
            # If absolute: clamp to the final expert target
            # If delta: mapping back from clamped interp_state to clamped interp_target
            # results in an identity matrix (zero delta), which is physically correct.
            final_target_params = expert_targets[-1].parameters()
            
            if action_space == "absolute":
                new_act_params = interp_action_se3.parameters()
                new_act_params[goal_indices] = final_target_params
                interp_action_se3 = vtf.SE3(new_act_params)
            else:
                # For delta space, ensure action is exactly identity at the goal
                new_act_mats = interp_action_se3.as_matrix()
                new_act_mats[goal_indices] = np.eye(4)
                interp_action_se3 = vtf.SE3.from_matrix(new_act_mats)
            
            interp_action_gripper[goal_indices] = act_gripper[-1]

    else:
        interp_action_se3 = None
        interp_action_gripper = None

    return interp_state_se3, interp_state_gripper, interp_timesteps, interp_action_se3, interp_action_gripper

# Pre-indexed version (amortised upfront cost)

class SpatialTrajectoryIndex:
    """Pre-computes trajectory backbone once for fast repeated queries.

    Use this when you will query the *same* trajectory at many different
    ``query_idx`` values (e.g. scrubbing through a visualisation).  The
    expensive SE3 log-map and cumulative-distance computation is done once
    in ``__init__``; each ``query()`` then only does the cheap interpolation.
    """

    def __init__(
        self,
        state_se3: vtf.SE3,
        state_gripper: np.ndarray,
        alpha: float = 0.01,
        action_se3: vtf.SE3 | None = None,
        action_gripper: np.ndarray | None = None,
        action_space: Literal["absolute", "delta"] = "delta",
    ) -> None:
        self.alpha = alpha
        self.T = len(state_se3.parameters())
        self.mats = state_se3.as_matrix()  # (T, 4, 4)
        self.action_space = action_space

        # Store gripper for interpolation
        self.gripper = state_gripper.astype(np.float64)
        self._gripper_1d = state_gripper.ndim == 1
        assert len(self.gripper) == self.T, (
            f"Gripper length ({len(self.gripper)}) must match poses ({self.T})"
        )

        # Vectorised relative transforms & log maps
        inv_mats = state_se3.inverse().as_matrix()
        rel_mats = np.einsum("nij,njk->nik", inv_mats[:-1], self.mats[1:])
        self.deltas = vtf.SE3.from_matrix(rel_mats).log()  # (T-1, 6)

        # Cumulative arc-length
        trans_norms = np.linalg.norm(self.deltas[:, :3], axis=-1)
        rot_norms = np.linalg.norm(self.deltas[:, 3:], axis=-1)
        step_dists = np.sqrt(trans_norms**2 + alpha * rot_norms**2)
        self.cumulative_dist = np.concatenate(([0.0], np.cumsum(step_dists)))
        self.total_dist = self.cumulative_dist[-1]

        # ── Pre-compute action backbone (if provided) ─────────────────
        self.has_actions = action_se3 is not None
        if self.has_actions:
            assert len(action_se3.parameters()) == self.T, (
                f"action_se3 length ({len(action_se3.parameters())}) must match state ({self.T})"
            )
            assert action_gripper is not None, (
                "action_gripper required when action_se3 is provided"
            )
            assert len(action_gripper) == self.T, (
                f"action_gripper length ({len(action_gripper)}) must match state ({self.T})"
            )

            self.act_gripper = action_gripper.astype(np.float64)
            self._act_gripper_1d = action_gripper.ndim == 1

            # Lift to absolute target space
            if action_space == "delta":
                self.target_mats = np.einsum(
                    "nij,njk->nik", self.mats, action_se3.as_matrix()
                )  # G_t = P_t * A_t
            else:
                self.target_mats = action_se3.as_matrix()

            # Consecutive target deltas for ScLERP
            target_inv_mats = vtf.SE3.from_matrix(self.target_mats).inverse().as_matrix()
            self.target_deltas = vtf.SE3.from_matrix(
                np.einsum("nij,njk->nik", target_inv_mats[:-1], self.target_mats[1:])
            ).log()  # (T-1, 6)

            # Final target for goal extrapolation
            self._final_target_params = vtf.SE3.from_matrix(
                self.target_mats[-1:]
            ).parameters()  # (1, 7)
        else:
            self.act_gripper = None
            self._act_gripper_1d = False
            self.target_mats = None
            self.target_deltas = None
            self._final_target_params = None

    def query(
        self,
        query_idx: int,
        step_size: float,
        horizon: int,
    ) -> Tuple[vtf.SE3, np.ndarray, np.ndarray, vtf.SE3 | None, np.ndarray | None]:
        """Return spatial action chunk at *query_idx*.

        Args:
            query_idx: Current time-index into the trajectory.
            step_size: Fixed Lie-norm distance between spatial keyframes.
            horizon: Number of spatial keyframes to return.

        Returns:
            Same as :func:`spatial_action_chunk_SE3`.
        """
        assert 0 <= query_idx < len(self.mats)
        assert horizon > 0 and step_size > 0

        current_s = self.cumulative_dist[query_idx]
        target_s = current_s + np.arange(1, horizon + 1) * step_size
        s_clamped = np.minimum(target_s, self.total_dist)

        indices = np.searchsorted(self.cumulative_dist, s_clamped) - 1
        indices = np.clip(indices, 0, len(self.deltas) - 1)

        seg_starts = self.cumulative_dist[indices]
        seg_ends = self.cumulative_dist[indices + 1]
        denom = seg_ends - seg_starts
        lerp_t = np.where(denom < 1e-15, 0.0, (s_clamped - seg_starts) / denom)

        interp_twists = lerp_t[:, None] * self.deltas[indices]
        delta_poses = vtf.SE3.exp(interp_twists)

        base_mats = self.mats[indices]
        res_mats = np.einsum("nij,njk->nik", base_mats, delta_poses.as_matrix())
        interp_state_se3 = vtf.SE3.from_matrix(res_mats)
        interp_timesteps = indices.astype(np.float64) + lerp_t

        # if trajectory end is reached, keep interp timesteps counting up
        at_goal_mask = target_s > self.total_dist
        if np.any(at_goal_mask):
            first_goal_idx = np.argmax(at_goal_mask)
            steps_past_goal = np.arange(len(interp_timesteps)) - first_goal_idx
            interp_timesteps[at_goal_mask] = (
                interp_timesteps[first_goal_idx] + steps_past_goal[at_goal_mask]
            )

        # Linearly interpolate state gripper
        next_indices = np.minimum(indices + 1, self.T - 1)
        interp_gripper = (
            (1.0 - lerp_t)[..., None] * self.gripper[indices]
            + lerp_t[..., None] * self.gripper[next_indices]
        )
        if self._gripper_1d:
            interp_gripper = interp_gripper.squeeze(-1)

        # ── Action interpolation ──────────────────────────────────────
        if self.has_actions:
            if self.action_space == "absolute":
                raise NotImplementedError("Absolute action space is supported but not yet tested / verified. Recommend visualizing as sanity check.")
            # ScLERP the absolute target trajectory
            interp_target_twists = lerp_t[:, None] * self.target_deltas[indices]
            delta_target_poses = vtf.SE3.exp(interp_target_twists)
            res_target_mats = np.einsum(
                "nij,njk->nik",
                self.target_mats[indices],
                delta_target_poses.as_matrix(),
            )
            interp_abs_targets = vtf.SE3.from_matrix(res_target_mats)

            # Project back to the requested action space
            if self.action_space == "delta":
                interp_action_se3 = vtf.SE3.from_matrix(
                    np.einsum(
                        "nij,njk->nik",
                        interp_state_se3.inverse().as_matrix(),
                        interp_abs_targets.as_matrix(),
                    )
                )
            else:
                interp_action_se3 = interp_abs_targets

            # Linearly interpolate action gripper
            interp_act_grip = (
                (1.0 - lerp_t)[..., None] * self.act_gripper[indices]
                + lerp_t[..., None] * self.act_gripper[next_indices]
            )
            if self._act_gripper_1d:
                interp_act_grip = interp_act_grip.squeeze(-1)

            # Goal extrapolation
            if np.any(at_goal_mask):
                goal_indices = np.where(at_goal_mask)[0]
                if self.action_space == "absolute":
                    new_params = interp_action_se3.parameters()
                    new_params[goal_indices] = self._final_target_params
                    interp_action_se3 = vtf.SE3(new_params)
                else:
                    new_mats = interp_action_se3.as_matrix()
                    new_mats[goal_indices] = np.eye(4)
                    interp_action_se3 = vtf.SE3.from_matrix(new_mats)
                interp_act_grip[goal_indices] = self.act_gripper[-1]
        else:
            interp_action_se3 = None
            interp_act_grip = None

        return interp_state_se3, interp_gripper, interp_timesteps, interp_action_se3, interp_act_grip
