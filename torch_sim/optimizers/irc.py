"""IRC (Intrinsic Reaction Coordinate) optimizer.

Implements the IRC algorithm following SCINE's IrcOptimizer:
1. Displace the TS geometry along the imaginary frequency mode
2. Run mass-weighted steepest descent optimization
3. Produce forward and backward IRC endpoints

Forward and backward IRC paths for all systems are run simultaneously
as a single batch of 2*n_systems systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from torch_sim.state import SimState

if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface


@dataclass
class IRCSettings:
    """Settings for the IRC optimizer.

    Mirrors SCINE's IrcOptimizer + SteepestDescent + GradientBasedCheck settings.
    """

    sd_factor: float = 2.0
    initial_step_size: float = 0.3
    max_iter: int = 100
    step_max_coeff: float = 2.0e-3
    step_rms: float = 1.0e-3
    grad_max_coeff: float = 2.0e-4
    grad_rms: float = 1.0e-4
    delta_value: float = 1.0e-6
    convergence_requirement: int = 3


def gradient_based_converged(
    gradients: torch.Tensor,
    step: torch.Tensor,
    delta_e: float,
    grad_max_coeff: float,
    grad_rms: float,
    step_max_coeff: float,
    step_rms: float,
    delta_value: float,
    requirement: int,
) -> bool:
    """SCINE-compatible GradientBasedCheck convergence test.

    Returns True when ``|delta_e| < delta_value`` AND at least *requirement*
    of the four gradient/step criteria are satisfied.
    """
    n_crit = 0
    if gradients.abs().max().item() < grad_max_coeff:
        n_crit += 1
    if step.abs().max().item() < step_max_coeff:
        n_crit += 1
    n_dof = gradients.numel()
    if n_dof > 0:
        if (gradients.norm() / (n_dof**0.5)).item() < grad_rms:
            n_crit += 1
        if (step.norm() / (n_dof**0.5)).item() < step_rms:
            n_crit += 1
    return abs(delta_e) < delta_value and n_crit >= requirement


def batch_irc_optimize(
    model: ModelInterface,
    state: SimState,
    modes: list[torch.Tensor],
    settings: IRCSettings | None = None,
) -> tuple[SimState, SimState, list[int], list[int]]:
    """Run IRC forward and backward for all systems in a batch.

    All 2*n_systems IRC paths (forward + backward for each system) run
    simultaneously with a single batched model call per SD step.

    Args:
        model: ModelInterface for energy/forces.
        state: Batched SimState with n_systems TS structures.
        modes: Per-system imaginary-frequency eigenvectors, each of shape
            ``[3*n_atoms_i]`` or ``[n_atoms_i, 3]``.
        settings: IRC settings (defaults if None).

    Returns:
        Tuple of ``(forward_state, backward_state, fwd_cycles, bwd_cycles)``.
    """
    settings = settings or IRCSettings()
    n_systems = state.n_systems
    device = state.device
    dtype = state.dtype

    counts = state.n_atoms_per_system
    offsets = torch.zeros(n_systems + 1, device=device, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)

    # --- 1. Displace along mode -----------------------------------------
    fwd_positions = state.positions.clone()
    bwd_positions = state.positions.clone()

    for s in range(n_systems):
        n_at = int(counts[s].item())
        off = int(offsets[s].item())
        mode = modes[s].to(device=device, dtype=dtype)
        if mode.dim() == 1:
            mode = mode.reshape(n_at, 3)
        max_val = mode.abs().max()
        if max_val > 0:
            scaled = settings.initial_step_size * mode / max_val
        else:
            scaled = torch.zeros_like(mode)
        fwd_positions[off : off + n_at] += scaled
        bwd_positions[off : off + n_at] -= scaled

    # --- 2. Build 2*n_systems batched state -----------------------------
    all_positions = torch.cat([fwd_positions, bwd_positions])
    all_masses = state.masses.repeat(2)
    all_atomic_numbers = state.atomic_numbers.repeat(2)
    all_cell = state.cell.repeat(2, 1, 1)
    all_sys_idx = torch.cat([state.system_idx, state.system_idx + n_systems])

    sqrt_masses = torch.sqrt(all_masses).unsqueeze(-1)  # [total_atoms, 1]

    # --- 3. Initial evaluation ------------------------------------------
    positions = all_positions.clone()

    def _make_state(pos: torch.Tensor) -> SimState:
        return SimState(
            positions=pos,
            masses=all_masses,
            cell=all_cell,
            pbc=state.pbc,
            atomic_numbers=all_atomic_numbers,
            system_idx=all_sys_idx,
        )

    out = model(_make_state(positions))
    energy = out["energy"]
    forces = out["forces"]

    total_systems = 2 * n_systems
    converged = torch.zeros(total_systems, dtype=torch.bool, device=device)
    cycle_counts = torch.zeros(total_systems, dtype=torch.long, device=device)

    # --- 4. Mass-weighted SD loop ---------------------------------------
    for cycle in range(1, settings.max_iter + 1):
        mw_forces = forces / sqrt_masses
        sd_step = settings.sd_factor * mw_forces

        old_positions = positions
        positions = positions + sd_step

        out = model(_make_state(positions))
        old_energy = energy
        energy = out["energy"]
        forces = out["forces"]

        delta_pos = positions - old_positions
        delta_e = energy - old_energy
        mw_grad = -forces / sqrt_masses

        for s in range(total_systems):
            if converged[s]:
                continue
            mask = all_sys_idx == s
            if gradient_based_converged(
                mw_grad[mask].reshape(-1),
                delta_pos[mask].reshape(-1),
                delta_e[s].item(),
                settings.grad_max_coeff,
                settings.grad_rms,
                settings.step_max_coeff,
                settings.step_rms,
                settings.delta_value,
                settings.convergence_requirement,
            ):
                converged[s] = True
                cycle_counts[s] = cycle

        if converged.all():
            break

    cycle_counts[~converged] = settings.max_iter

    # --- 5. Split forward / backward ------------------------------------
    n_atoms_total = state.positions.shape[0]

    fwd_state = SimState(
        positions=positions[:n_atoms_total].clone(),
        masses=state.masses.clone(),
        cell=state.cell.clone(),
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers.clone(),
        system_idx=state.system_idx.clone(),
    )
    bwd_state = SimState(
        positions=positions[n_atoms_total:].clone(),
        masses=state.masses.clone(),
        cell=state.cell.clone(),
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers.clone(),
        system_idx=state.system_idx.clone(),
    )

    return (
        fwd_state,
        bwd_state,
        cycle_counts[:n_systems].tolist(),
        cycle_counts[n_systems:].tolist(),
    )
