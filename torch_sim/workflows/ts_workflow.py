"""Transition-state workflow: NT2 -> TSOPT -> Hessian -> IRC -> IRCOPT.

Reimplements the readuct task-chaining logic from single_shot_rxn_delta
using torch-sim components.  All stages are batch-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from torch_sim.hessian import compute_hessian
from torch_sim.optimizers.bofill import BofillSettings, batch_bofill_ts_optimize
from torch_sim.optimizers.irc import (
    IRCSettings,
    batch_irc_optimize,
    gradient_based_converged,
)
from torch_sim.optimizers.nt2 import NT2Settings, batch_nt2_optimize
from torch_sim.state import SimState, concatenate_states

if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class GeomOptSettings:
    """Settings for BFGS geometry optimization (used by IRCOPT)."""

    max_iter: int = 1000
    bfgs_trust_radius: float = 0.2
    bfgs_alpha: float = 70.0
    step_max_coeff: float = 2.0e-3
    step_rms: float = 1.0e-3
    grad_max_coeff: float = 2.0e-4
    grad_rms: float = 1.0e-4
    delta_value: float = 1.0e-6
    convergence_requirement: int = 3


@dataclass
class TSWorkflowSettings:
    """Aggregated settings for the full TS workflow."""

    nt2: NT2Settings = field(default_factory=NT2Settings)
    tsopt: BofillSettings = field(default_factory=BofillSettings)
    tsopt_hessian_delta: float = 0.01
    hessian_delta: float = 0.01
    irc: IRCSettings = field(default_factory=IRCSettings)
    ircopt: GeomOptSettings = field(default_factory=GeomOptSettings)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TSWorkflowResult:
    """Results from the full TS workflow."""

    ts_guess: SimState
    ts_opt: SimState
    hessians: list[torch.Tensor]
    modes: list[torch.Tensor]
    eigenvalues: list[torch.Tensor]
    irc_forward: SimState
    irc_backward: SimState
    irc_forward_opt: SimState
    irc_backward_opt: SimState
    nt2_trajectories: list[list[torch.Tensor]]
    nt2_energies: list[list[float]]
    tsopt_cycles: list[int]
    irc_fwd_cycles: list[int]
    irc_bwd_cycles: list[int]
    ircopt_fwd_cycles: list[int]
    ircopt_bwd_cycles: list[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_system(state: SimState, sys_idx: int) -> SimState:
    """Extract a single system from a batched SimState."""
    counts = state.n_atoms_per_system
    offsets = torch.zeros(state.n_systems + 1, device=state.device, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)
    n_at = int(counts[sys_idx].item())
    off = int(offsets[sys_idx].item())
    return SimState(
        positions=state.positions[off : off + n_at].clone(),
        masses=state.masses[off : off + n_at].clone(),
        cell=state.cell[sys_idx : sys_idx + 1].clone(),
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers[off : off + n_at].clone(),
    )


# ---------------------------------------------------------------------------
# Geometry optimization (IRCOPT)
# ---------------------------------------------------------------------------


def batch_geometry_optimize(
    model: ModelInterface,
    state: SimState,
    settings: GeomOptSettings | None = None,
) -> tuple[SimState, list[int]]:
    """BFGS geometry optimization with SCINE-style convergence.

    Runs the torch-sim BFGS optimizer on a batched SimState, checking
    convergence per system using :func:`gradient_based_converged`.

    Args:
        model: ModelInterface for energy/forces.
        state: Batched SimState (one or more systems).
        settings: Optimization settings.

    Returns:
        ``(optimized_state, cycle_counts_per_system)``
    """
    from torch_sim.optimizers.bfgs import bfgs_init, bfgs_step

    settings = settings or GeomOptSettings()

    bfgs_state = bfgs_init(
        state,
        model,
        max_step=settings.bfgs_trust_radius,
        alpha=settings.bfgs_alpha,
    )

    n_systems = state.n_systems
    device = state.device
    converged = torch.zeros(n_systems, dtype=torch.bool, device=device)
    cycle_counts = torch.zeros(n_systems, dtype=torch.long, device=device)

    prev_energy = bfgs_state.energy.clone()

    for cycle in range(1, settings.max_iter + 1):
        prev_positions = bfgs_state.positions.clone()

        bfgs_state = bfgs_step(bfgs_state, model)

        step = bfgs_state.positions - prev_positions
        delta_e = bfgs_state.energy - prev_energy
        prev_energy = bfgs_state.energy.clone()
        grad = -bfgs_state.forces

        for s in range(n_systems):
            if converged[s]:
                continue
            mask = bfgs_state.system_idx == s
            if gradient_based_converged(
                grad[mask].reshape(-1),
                step[mask].reshape(-1),
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

    result = SimState(
        positions=bfgs_state.positions.clone(),
        masses=bfgs_state.masses.clone(),
        cell=bfgs_state.cell.clone(),
        pbc=bfgs_state.pbc,
        atomic_numbers=bfgs_state.atomic_numbers.clone(),
        system_idx=bfgs_state.system_idx.clone(),
    )
    return result, cycle_counts.tolist()


# ---------------------------------------------------------------------------
# Full workflow
# ---------------------------------------------------------------------------


def batch_ts_workflow(
    model: ModelInterface,
    state: SimState,
    association_lists: list[list[int]],
    dissociation_lists: list[list[int]],
    settings: TSWorkflowSettings | None = None,
) -> TSWorkflowResult:
    """Run the full NT2 -> TSOPT -> Hessian -> IRC -> IRCOPT workflow.

    All stages operate on the full batch of systems.  The model is called
    in batched mode wherever possible.

    Args:
        model: ModelInterface for energy/forces (must support batching).
        state: Batched SimState with n_systems reactant geometries.
        association_lists: Per-system association index pairs for NT2.
        dissociation_lists: Per-system dissociation index pairs for NT2.
        settings: Workflow settings (defaults if None).

    Returns:
        :class:`TSWorkflowResult` with all intermediate and final states.
    """
    settings = settings or TSWorkflowSettings()
    n_systems = state.n_systems

    # --- Step 1: NT2 -> TS guess ----------------------------------------
    nt2_settings = [settings.nt2] * n_systems
    ts_guess, nt2_trajectories, nt2_energies = batch_nt2_optimize(
        model,
        state,
        association_lists,
        dissociation_lists,
        settings_list=nt2_settings,
    )

    # --- Step 2: TSOPT (Bofill) -> optimized TS -------------------------
    tsopt_settings = [settings.tsopt] * n_systems
    ts_opt, tsopt_cycles = batch_bofill_ts_optimize(
        model,
        ts_guess,
        settings_list=tsopt_settings,
        hessian_delta=settings.tsopt_hessian_delta,
    )

    # --- Step 3: Hessian at optimized TS --------------------------------
    hessians: list[torch.Tensor] = []
    modes: list[torch.Tensor] = []
    eigenvalues_list: list[torch.Tensor] = []

    for s in range(n_systems):
        single = _extract_system(ts_opt, s)
        H = compute_hessian(model, single, delta=settings.hessian_delta)
        evals, evecs = torch.linalg.eigh(H)
        hessians.append(H)
        eigenvalues_list.append(evals)
        modes.append(evecs[:, 0])

    # --- Step 4: IRC -> forward/backward endpoints ----------------------
    irc_fwd, irc_bwd, irc_fwd_cycles, irc_bwd_cycles = batch_irc_optimize(
        model,
        ts_opt,
        modes,
        settings=settings.irc,
    )

    # --- Step 5: IRCOPT -> optimized endpoints --------------------------
    irc_combined = concatenate_states([irc_fwd, irc_bwd])
    ircopt_combined, ircopt_cycles = batch_geometry_optimize(
        model,
        irc_combined,
        settings=settings.ircopt,
    )

    n_atoms_fwd = irc_fwd.positions.shape[0]
    irc_fwd_opt = SimState(
        positions=ircopt_combined.positions[:n_atoms_fwd].clone(),
        masses=irc_fwd.masses.clone(),
        cell=irc_fwd.cell.clone(),
        pbc=irc_fwd.pbc,
        atomic_numbers=irc_fwd.atomic_numbers.clone(),
        system_idx=irc_fwd.system_idx.clone(),
    )
    irc_bwd_opt = SimState(
        positions=ircopt_combined.positions[n_atoms_fwd:].clone(),
        masses=irc_bwd.masses.clone(),
        cell=irc_bwd.cell.clone(),
        pbc=irc_bwd.pbc,
        atomic_numbers=irc_bwd.atomic_numbers.clone(),
        system_idx=irc_bwd.system_idx.clone(),
    )

    ircopt_fwd_cycles = ircopt_cycles[:n_systems]
    ircopt_bwd_cycles = ircopt_cycles[n_systems:]

    return TSWorkflowResult(
        ts_guess=ts_guess,
        ts_opt=ts_opt,
        hessians=hessians,
        modes=modes,
        eigenvalues=eigenvalues_list,
        irc_forward=irc_fwd,
        irc_backward=irc_bwd,
        irc_forward_opt=irc_fwd_opt,
        irc_backward_opt=irc_bwd_opt,
        nt2_trajectories=nt2_trajectories,
        nt2_energies=nt2_energies,
        tsopt_cycles=tsopt_cycles,
        irc_fwd_cycles=irc_fwd_cycles,
        irc_bwd_cycles=irc_bwd_cycles,
        ircopt_fwd_cycles=ircopt_fwd_cycles,
        ircopt_bwd_cycles=ircopt_bwd_cycles,
    )
