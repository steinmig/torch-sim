"""Transition-state workflow: NT2 -> TSOPT -> Hessian -> IRC -> IRCOPT.

Reimplements the readuct task-chaining logic from single_shot_rxn_delta
using torch-sim components.  All stages are batch-compatible.

Includes a memory-safe variant (``memory_safe_ts_workflow``) that processes
arbitrarily many reaction coordinates by splitting them into GPU-safe
sub-batches at each stage.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from torch_sim.hessian import compute_hessian
from torch_sim.io import state_to_atoms
from torch_sim.optimizers.bofill import BofillSettings, batch_bofill_ts_optimize
from torch_sim.optimizers.dimer import DimerSettings, batch_dimer_ts_optimize
from torch_sim.optimizers.irc import (
    IRCSettings,
    batch_irc_optimize,
    gradient_based_converged,
)
from torch_sim.optimizers.nt2 import NT2Settings, batch_nt2_optimize
from torch_sim.state import SimState, concatenate_states

if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface

logger = logging.getLogger(__name__)

# SCINE uses Bohr / Hartree.  torch-sim uses Angstrom / eV.
_BOHR_TO_ANG = 0.529177
_HARTREE_TO_EV = 27.2114
_HA_BOHR_TO_EV_ANG = _HARTREE_TO_EV / _BOHR_TO_ANG  # ~51.422 eV/A per Ha/Bohr


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class GeomOptSettings:
    """Settings for BFGS geometry optimization (Angstrom / eV units).

    All defaults converted from SCINE's GradientBasedCheck atomic units.
    """

    max_iter: int = 1000
    bfgs_trust_radius: float = 0.2 * _BOHR_TO_ANG  # ~0.106 A
    bfgs_alpha: float = 70.0
    step_max_coeff: float = 2.0e-3 * _BOHR_TO_ANG  # ~1.06e-3 A
    step_rms: float = 1.0e-3 * _BOHR_TO_ANG  # ~5.29e-4 A
    grad_max_coeff: float = 2.0e-4 * _HA_BOHR_TO_EV_ANG  # ~1.03e-2 eV/A
    grad_rms: float = 1.0e-4 * _HA_BOHR_TO_EV_ANG  # ~5.14e-3 eV/A
    delta_value: float = 1.0e-6 * _HARTREE_TO_EV  # ~2.72e-5 eV
    convergence_requirement: int = 3


@dataclass
class TSWorkflowSettings:
    """Aggregated settings for the full TS workflow."""

    nt2: NT2Settings = field(default_factory=NT2Settings)
    tsopt_method: str = "bofill"
    tsopt: BofillSettings = field(default_factory=BofillSettings)
    tsopt_dimer: DimerSettings = field(default_factory=DimerSettings)
    tsopt_hessian_delta: float = 0.01
    hessian_delta: float = 0.01
    irc: IRCSettings = field(default_factory=IRCSettings)
    ircopt: GeomOptSettings = field(default_factory=GeomOptSettings)
    output_dir: str | None = None


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
# I/O helpers
# ---------------------------------------------------------------------------


def _write_state(state: SimState, path: Path, label: str) -> None:
    """Write every system in *state* to an XYZ file at *path*."""
    from ase.io import write

    atoms_list = state_to_atoms(state)
    for i, atoms in enumerate(atoms_list):
        atoms.info["label"] = f"{label}_sys{i}"
    write(str(path), atoms_list, format="extxyz")
    logger.info("  Wrote %s (%d systems) -> %s", label, len(atoms_list), path)


def _write_nt2_trajectories(
    state: SimState,
    trajectories: list[list[torch.Tensor]],
    energies: list[list[float]],
    output_dir: Path,
) -> None:
    """Write per-system NT2 trajectory XYZ files."""
    from ase.io import write

    ref_atoms_list = state_to_atoms(state)
    for s, (traj_positions, traj_energies) in enumerate(
        zip(trajectories, energies)
    ):
        if not traj_positions:
            continue
        ref = ref_atoms_list[min(s, len(ref_atoms_list) - 1)]
        frames = []
        for step, (pos_flat, e) in enumerate(zip(traj_positions, traj_energies)):
            from ase import Atoms

            atoms = Atoms(
                numbers=ref.get_atomic_numbers(),
                positions=pos_flat.detach().cpu().numpy().reshape(-1, 3),
                cell=ref.get_cell(),
                pbc=ref.pbc,
            )
            atoms.info["energy"] = e
            atoms.info["step"] = step
            frames.append(atoms)
        path = output_dir / f"nt2_trajectory_sys{s}.xyz"
        write(str(path), frames, format="extxyz")
        logger.info("  Wrote NT2 trajectory sys %d (%d frames) -> %s", s, len(frames), path)


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


def _extract_range(state: SimState, start: int, end: int) -> SimState:
    """Extract systems [start, end) from a batched SimState."""
    return state[list(range(start, end))]


# ---------------------------------------------------------------------------
# Sub-batching utilities
# ---------------------------------------------------------------------------


def _chunk_indices(n_total: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` pairs that cover ``[0, n_total)``."""
    return [
        (i, min(i + chunk_size, n_total))
        for i in range(0, n_total, chunk_size)
    ]


def estimate_max_systems(
    model: ModelInterface,
    template_state: SimState,
    *,
    memory_scales_with: str = "n_atoms",
    memory_scaling_factor: float = 1.6,
    max_memory_padding: float = 0.8,
) -> int:
    """Probe GPU memory to determine how many copies of *template_state* fit.

    Uses :class:`~torch_sim.autobatching.BinningAutoBatcher` internally to
    estimate ``max_memory_scaler``, then derives ``max_systems`` assuming all
    systems have the same atom count.

    Args:
        model: ModelInterface used for the workflow.
        template_state: A single-system SimState representative of the systems
            to be processed.
        memory_scales_with: Memory scaling metric (``"n_atoms"`` recommended
            for homogeneous batches).
        memory_scaling_factor: Geometric growth factor for the GPU probing.
        max_memory_padding: Safety factor applied to the estimated limit
            (0.8 = use 80 % of estimated capacity).

    Returns:
        Maximum number of systems that fit in one batch.
    """
    from torch_sim.autobatching import (
        BinningAutoBatcher,
        calculate_memory_scalers,
    )

    if template_state.n_systems != 1:
        template_state = _extract_system(template_state, 0)

    batcher = BinningAutoBatcher(
        model,
        memory_scales_with=memory_scales_with,
        memory_scaling_factor=memory_scaling_factor,
        max_memory_padding=max_memory_padding,
    )
    probe = concatenate_states([template_state] * 2)
    max_scaler = batcher.load_states(probe)

    per_system_scalers = calculate_memory_scalers(
        template_state, memory_scales_with=memory_scales_with,
    )
    per_system = per_system_scalers[0]
    if per_system <= 0:
        return 1
    return max(1, int(max_scaler // per_system))


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

    out_dir: Path | None = None
    if settings.output_dir is not None:
        out_dir = Path(settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Workflow output directory: %s", out_dir)

    # --- Step 1: NT2 -> TS guess ----------------------------------------
    logger.info(
        "[Step 1/5] NT2 scan — searching for TS guesses (%d systems)", n_systems
    )
    nt2_settings = [settings.nt2] * n_systems
    ts_guess, nt2_trajectories, nt2_energies = batch_nt2_optimize(
        model,
        state,
        association_lists,
        dissociation_lists,
        settings_list=nt2_settings,
    )
    logger.info(
        "[Step 1/5] NT2 complete — TS guesses obtained for %d systems", n_systems
    )
    if out_dir is not None:
        _write_state(ts_guess, out_dir / "ts_guess.xyz", "ts_guess")
        _write_nt2_trajectories(
            state, nt2_trajectories, nt2_energies, out_dir
        )

    # --- Step 2: TSOPT -> optimized TS -----------------------------------
    if settings.tsopt_method == "dimer":
        logger.info("[Step 2/5] TSOPT (Dimer) — optimizing TS guesses")
        dimer_settings = [settings.tsopt_dimer] * n_systems
        ts_opt, tsopt_cycles = batch_dimer_ts_optimize(
            model,
            ts_guess,
            settings_list=dimer_settings,
        )
    else:
        logger.info("[Step 2/5] TSOPT (Bofill) — optimizing TS guesses")
        tsopt_settings = [settings.tsopt] * n_systems
        ts_opt, tsopt_cycles = batch_bofill_ts_optimize(
            model,
            ts_guess,
            settings_list=tsopt_settings,
            hessian_delta=settings.tsopt_hessian_delta,
        )
    logger.info(
        "[Step 2/5] TSOPT complete — cycles per system: %s", tsopt_cycles
    )
    if out_dir is not None:
        _write_state(ts_opt, out_dir / "ts_opt.xyz", "ts_opt")

    # --- Step 3: Hessian at optimized TS --------------------------------
    logger.info("[Step 3/5] Hessian — computing at optimized TS geometries")
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
        logger.info(
            "  System %d: lowest eigenvalue = %.6f", s, evals[0].item()
        )

    n_imaginary = sum(1 for ev in eigenvalues_list if ev[0] < 0)
    logger.info(
        "[Step 3/5] Hessian complete — %d/%d systems have imaginary frequency",
        n_imaginary,
        n_systems,
    )

    # --- Step 4: IRC -> forward/backward endpoints ----------------------
    logger.info("[Step 4/5] IRC — following reaction path from TS")
    irc_fwd, irc_bwd, irc_fwd_cycles, irc_bwd_cycles = batch_irc_optimize(
        model,
        ts_opt,
        modes,
        settings=settings.irc,
    )
    logger.info(
        "[Step 4/5] IRC complete — forward cycles: %s, backward cycles: %s",
        irc_fwd_cycles,
        irc_bwd_cycles,
    )
    if out_dir is not None:
        _write_state(irc_fwd, out_dir / "irc_forward.xyz", "irc_forward")
        _write_state(irc_bwd, out_dir / "irc_backward.xyz", "irc_backward")

    # --- Step 5: IRCOPT -> optimized endpoints --------------------------
    logger.info("[Step 5/5] IRCOPT (BFGS) — optimizing IRC endpoints")
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
    logger.info(
        "[Step 5/5] IRCOPT complete — forward cycles: %s, backward cycles: %s",
        ircopt_fwd_cycles,
        ircopt_bwd_cycles,
    )
    if out_dir is not None:
        _write_state(
            irc_fwd_opt, out_dir / "irc_forward_opt.xyz", "irc_forward_opt"
        )
        _write_state(
            irc_bwd_opt, out_dir / "irc_backward_opt.xyz", "irc_backward_opt"
        )

    logger.info("TS workflow finished for %d systems", n_systems)

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


# ---------------------------------------------------------------------------
# Memory-safe workflow
# ---------------------------------------------------------------------------


def memory_safe_ts_workflow(
    model: ModelInterface,
    state: SimState,
    association_lists: list[list[int]],
    dissociation_lists: list[list[int]],
    settings: TSWorkflowSettings | None = None,
    max_systems: int | None = None,
) -> TSWorkflowResult:
    """Run the full TS workflow with automatic memory-safe sub-batching.

    Identical to :func:`batch_ts_workflow` but splits the *N* systems into
    GPU-safe sub-batches at **each stage**, using different batch budgets
    per stage (e.g. IRC needs 0.5x because it doubles the systems).

    Between-stage filtering removes systems that:
    - After Hessian: have no imaginary frequency (eigenvalue[0] >= 0).

    Args:
        model: ModelInterface (must support batching).
        state: Batched SimState with *N* reactant geometries.
        association_lists: Per-system association atom-index pairs for NT2.
        dissociation_lists: Per-system dissociation atom-index pairs for NT2.
        settings: Workflow settings.
        max_systems: Maximum number of systems per sub-batch.  If *None*,
            auto-detected via GPU memory probing.

    Returns:
        :class:`TSWorkflowResult` with results for all surviving systems.
    """
    settings = settings or TSWorkflowSettings()
    n_total = state.n_systems

    out_dir: Path | None = None
    if settings.output_dir is not None:
        out_dir = Path(settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Workflow output directory: %s", out_dir)

    # --- Determine batch budget -------------------------------------------
    if max_systems is None:
        logger.info("Probing GPU memory to determine max_systems ...")
        max_systems = estimate_max_systems(model, _extract_system(state, 0))
        logger.info("  Auto-detected max_systems = %d", max_systems)
    else:
        logger.info("Using user-specified max_systems = %d", max_systems)

    irc_budget = max(1, max_systems // 2)

    # =====================================================================
    # Stage 1: NT2  (budget: max_systems)
    # =====================================================================
    logger.info(
        "[Step 1/5] NT2 scan — %d systems, chunk size %d", n_total, max_systems,
    )
    all_ts_guesses: list[SimState] = []
    all_nt2_traj: list[list[torch.Tensor]] = []
    all_nt2_energies: list[list[float]] = []

    for start, end in _chunk_indices(n_total, max_systems):
        chunk_state = _extract_range(state, start, end)
        chunk_assoc = association_lists[start:end]
        chunk_dissoc = dissociation_lists[start:end]
        n_chunk = end - start
        logger.info("  NT2 chunk [%d:%d] (%d systems)", start, end, n_chunk)

        ts_guess, traj, energies = batch_nt2_optimize(
            model,
            chunk_state,
            chunk_assoc,
            chunk_dissoc,
            settings_list=[settings.nt2] * n_chunk,
        )
        all_ts_guesses.append(ts_guess)
        all_nt2_traj.extend(traj)
        all_nt2_energies.extend(energies)

    ts_guess_all = concatenate_states(all_ts_guesses)
    n_after_nt2 = ts_guess_all.n_systems
    logger.info("[Step 1/5] NT2 complete — %d TS guesses", n_after_nt2)

    if out_dir is not None:
        _write_state(ts_guess_all, out_dir / "ts_guess.xyz", "ts_guess")
        _write_nt2_trajectories(state, all_nt2_traj, all_nt2_energies, out_dir)

    # =====================================================================
    # Stage 2: TSOPT  (budget: max_systems)
    # =====================================================================
    logger.info(
        "[Step 2/5] TSOPT (%s) — %d systems, chunk size %d",
        settings.tsopt_method, n_after_nt2, max_systems,
    )
    all_ts_opt: list[SimState] = []
    all_tsopt_cycles: list[int] = []

    for start, end in _chunk_indices(n_after_nt2, max_systems):
        chunk = _extract_range(ts_guess_all, start, end)
        n_chunk = end - start
        logger.info("  TSOPT chunk [%d:%d] (%d systems)", start, end, n_chunk)

        if settings.tsopt_method == "dimer":
            ts_opt_chunk, cycles = batch_dimer_ts_optimize(
                model, chunk,
                settings_list=[settings.tsopt_dimer] * n_chunk,
            )
        else:
            ts_opt_chunk, cycles = batch_bofill_ts_optimize(
                model, chunk,
                settings_list=[settings.tsopt] * n_chunk,
                hessian_delta=settings.tsopt_hessian_delta,
            )
        all_ts_opt.append(ts_opt_chunk)
        all_tsopt_cycles.extend(cycles)

    ts_opt_all = concatenate_states(all_ts_opt)
    logger.info("[Step 2/5] TSOPT complete — cycles: %s", all_tsopt_cycles)

    if out_dir is not None:
        _write_state(ts_opt_all, out_dir / "ts_opt.xyz", "ts_opt")

    # =====================================================================
    # Stage 3: Hessian  (per-system, no batching needed)
    # =====================================================================
    logger.info("[Step 3/5] Hessian — %d systems (per-system)", n_after_nt2)
    all_hessians: list[torch.Tensor] = []
    all_modes: list[torch.Tensor] = []
    all_eigenvalues: list[torch.Tensor] = []

    for s in range(n_after_nt2):
        single = _extract_system(ts_opt_all, s)
        H = compute_hessian(
            model, single, delta=settings.hessian_delta,
            batch_perturbations=True,
        )
        evals, evecs = torch.linalg.eigh(H)
        all_hessians.append(H)
        all_eigenvalues.append(evals)
        all_modes.append(evecs[:, 0])
        logger.info("  System %d: lowest eigenvalue = %.6f", s, evals[0].item())

    n_imaginary = sum(1 for ev in all_eigenvalues if ev[0] < 0)
    logger.info(
        "[Step 3/5] Hessian complete — %d/%d have imaginary frequency",
        n_imaginary, n_after_nt2,
    )

    # --- Filter: keep only systems with imaginary frequency ---------------
    keep_idx = [i for i, ev in enumerate(all_eigenvalues) if ev[0] < 0]
    n_filtered = n_after_nt2 - len(keep_idx)
    if n_filtered > 0:
        logger.info(
            "  Filtering: dropping %d systems without imaginary frequency",
            n_filtered,
        )

    if not keep_idx:
        logger.warning("No systems have imaginary frequency — workflow stops.")
        empty = SimState(
            positions=torch.zeros(0, 3, device=state.device, dtype=state.dtype),
            masses=torch.zeros(0, device=state.device, dtype=state.dtype),
            cell=torch.zeros(0, 3, 3, device=state.device, dtype=state.dtype),
            pbc=state.pbc,
            atomic_numbers=torch.zeros(0, device=state.device, dtype=torch.int32),
            system_idx=torch.zeros(0, device=state.device, dtype=torch.long),
        )
        return TSWorkflowResult(
            ts_guess=ts_guess_all,
            ts_opt=ts_opt_all,
            hessians=all_hessians,
            modes=all_modes,
            eigenvalues=all_eigenvalues,
            irc_forward=empty,
            irc_backward=empty,
            irc_forward_opt=empty,
            irc_backward_opt=empty,
            nt2_trajectories=all_nt2_traj,
            nt2_energies=all_nt2_energies,
            tsopt_cycles=all_tsopt_cycles,
            irc_fwd_cycles=[],
            irc_bwd_cycles=[],
            ircopt_fwd_cycles=[],
            ircopt_bwd_cycles=[],
        )

    ts_opt_filtered = ts_opt_all[keep_idx]
    modes_filtered = [all_modes[i] for i in keep_idx]
    n_irc = len(keep_idx)

    # =====================================================================
    # Stage 4: IRC  (budget: irc_budget because 2x internal expansion)
    # =====================================================================
    logger.info(
        "[Step 4/5] IRC — %d systems, chunk size %d (0.5x budget)",
        n_irc, irc_budget,
    )
    all_irc_fwd: list[SimState] = []
    all_irc_bwd: list[SimState] = []
    all_irc_fwd_cycles: list[int] = []
    all_irc_bwd_cycles: list[int] = []

    for start, end in _chunk_indices(n_irc, irc_budget):
        chunk = _extract_range(ts_opt_filtered, start, end)
        chunk_modes = modes_filtered[start:end]
        n_chunk = end - start
        logger.info("  IRC chunk [%d:%d] (%d systems)", start, end, n_chunk)

        fwd, bwd, fc, bc = batch_irc_optimize(
            model, chunk, chunk_modes, settings=settings.irc,
        )
        all_irc_fwd.append(fwd)
        all_irc_bwd.append(bwd)
        all_irc_fwd_cycles.extend(fc)
        all_irc_bwd_cycles.extend(bc)

    irc_fwd_all = concatenate_states(all_irc_fwd)
    irc_bwd_all = concatenate_states(all_irc_bwd)
    logger.info(
        "[Step 4/5] IRC complete — fwd cycles: %s, bwd cycles: %s",
        all_irc_fwd_cycles, all_irc_bwd_cycles,
    )
    if out_dir is not None:
        _write_state(irc_fwd_all, out_dir / "irc_forward.xyz", "irc_forward")
        _write_state(irc_bwd_all, out_dir / "irc_backward.xyz", "irc_backward")

    # =====================================================================
    # Stage 5: IRCOPT  (budget: irc_budget for fwd+bwd combined)
    # =====================================================================
    logger.info(
        "[Step 5/5] IRCOPT — %d fwd + %d bwd systems, chunk size %d",
        n_irc, n_irc, irc_budget,
    )
    all_ircopt_fwd: list[SimState] = []
    all_ircopt_bwd: list[SimState] = []
    all_ircopt_fwd_cycles: list[int] = []
    all_ircopt_bwd_cycles: list[int] = []

    for start, end in _chunk_indices(n_irc, irc_budget):
        fwd_chunk = _extract_range(irc_fwd_all, start, end)
        bwd_chunk = _extract_range(irc_bwd_all, start, end)
        combined = concatenate_states([fwd_chunk, bwd_chunk])
        n_chunk = end - start
        logger.info("  IRCOPT chunk [%d:%d] (%d systems x2)", start, end, n_chunk)

        opt_combined, opt_cycles = batch_geometry_optimize(
            model, combined, settings=settings.ircopt,
        )

        n_at_fwd = fwd_chunk.positions.shape[0]
        fwd_opt = SimState(
            positions=opt_combined.positions[:n_at_fwd].clone(),
            masses=fwd_chunk.masses.clone(),
            cell=fwd_chunk.cell.clone(),
            pbc=fwd_chunk.pbc,
            atomic_numbers=fwd_chunk.atomic_numbers.clone(),
            system_idx=fwd_chunk.system_idx.clone(),
        )
        bwd_opt = SimState(
            positions=opt_combined.positions[n_at_fwd:].clone(),
            masses=bwd_chunk.masses.clone(),
            cell=bwd_chunk.cell.clone(),
            pbc=bwd_chunk.pbc,
            atomic_numbers=bwd_chunk.atomic_numbers.clone(),
            system_idx=bwd_chunk.system_idx.clone(),
        )
        all_ircopt_fwd.append(fwd_opt)
        all_ircopt_bwd.append(bwd_opt)
        all_ircopt_fwd_cycles.extend(opt_cycles[:n_chunk])
        all_ircopt_bwd_cycles.extend(opt_cycles[n_chunk:])

    irc_fwd_opt_all = concatenate_states(all_ircopt_fwd)
    irc_bwd_opt_all = concatenate_states(all_ircopt_bwd)
    logger.info(
        "[Step 5/5] IRCOPT complete — fwd cycles: %s, bwd cycles: %s",
        all_ircopt_fwd_cycles, all_ircopt_bwd_cycles,
    )
    if out_dir is not None:
        _write_state(
            irc_fwd_opt_all, out_dir / "irc_forward_opt.xyz", "irc_forward_opt",
        )
        _write_state(
            irc_bwd_opt_all, out_dir / "irc_backward_opt.xyz", "irc_backward_opt",
        )

    logger.info(
        "Memory-safe TS workflow finished: %d total -> %d after Hessian filter",
        n_total, n_irc,
    )

    return TSWorkflowResult(
        ts_guess=ts_guess_all,
        ts_opt=ts_opt_all,
        hessians=all_hessians,
        modes=all_modes,
        eigenvalues=all_eigenvalues,
        irc_forward=irc_fwd_all,
        irc_backward=irc_bwd_all,
        irc_forward_opt=irc_fwd_opt_all,
        irc_backward_opt=irc_bwd_opt_all,
        nt2_trajectories=all_nt2_traj,
        nt2_energies=all_nt2_energies,
        tsopt_cycles=all_tsopt_cycles,
        irc_fwd_cycles=all_irc_fwd_cycles,
        irc_bwd_cycles=all_irc_bwd_cycles,
        ircopt_fwd_cycles=all_ircopt_fwd_cycles,
        ircopt_bwd_cycles=all_ircopt_bwd_cycles,
    )
