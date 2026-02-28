"""Bofill optimizer for saddle point (transition state) optimization.

Reimplements the SCINE Bofill algorithm: minimizes along all modes except
one, along which it maximizes. Uses Hessian updates (PSB + SR1 blend)
between full Hessian calculations.

Reference: Phys. Chem. Chem. Phys., 2002, 4, 11-15;
           J. Comput. Chem., 1994, 15, 1 (Bofill original).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from torch_sim.hessian import compute_hessian
from torch_sim.state import SimState

if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface


@dataclass
class BofillSettings:
    """Settings for the Bofill optimizer."""
    trust_radius: float = 0.1
    hessian_update: int = 5
    mode_to_follow: int = 0
    max_iter: int = 100
    step_max_coeff: float = 1e-4
    step_rms: float = 5e-4
    grad_max_coeff: float = 5e-5
    grad_rms: float = 1e-5
    delta_value: float = 1e-7
    convergence_requirement: int = 3
    max_value_memory: int = 10


def _is_oscillating(mem: deque) -> bool:
    """Check if stored values show an alternating pattern (SCINE convention)."""
    n = len(mem)
    if n < 3:
        return False
    if abs(mem[0] - mem[1]) < 1e-12:
        return False
    prev_positive = (mem[0] - mem[1]) > 0.0
    for i in range(2, n):
        this_positive = (mem[i - 1] - mem[i]) > 0.0
        if prev_positive == this_positive:
            return False
        prev_positive = this_positive
    return True


def _mode_maximization_with_hessian(
    gradients: torch.Tensor,
    hessian: torch.Tensor,
    mode_to_maximize: int,
    prev_eigenvector: torch.Tensor | None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Compute step that maximizes along one mode and minimizes along all others.

    Returns (steps, updated_mode_to_maximize, followed_eigenvector).
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    n = eigenvalues.shape[0]

    minstep = eigenvectors.T @ gradients

    if prev_eigenvector is not None:
        overlaps = torch.zeros(n, device=gradients.device, dtype=gradients.dtype)
        for i in range(n):
            if eigenvalues[i] >= 0.0:
                if i == 0:
                    raise RuntimeError("No negative eigenvalue found in Hessian")
                break
            cos_sim = (prev_eigenvector @ eigenvectors[:, i]) / (
                prev_eigenvector.norm() * eigenvectors[:, i].norm()
            )
            overlaps[i] = cos_sim

        mode_to_maximize = int(overlaps.abs().argmax().item())

    followed_ev = eigenvectors[:, mode_to_maximize].clone()

    # Lambda_P via 2x2 eigenproblem
    tmp1 = torch.tensor(
        [[eigenvalues[mode_to_maximize].item(), minstep[mode_to_maximize].item()],
         [minstep[mode_to_maximize].item(), 0.0]],
        device=gradients.device, dtype=gradients.dtype,
    )
    lambda_p = torch.linalg.eigvalsh(tmp1)[-1]

    # Collect non-followed eigenvalues/minsteps
    not_follow_idx = [i for i in range(n) if i != mode_to_maximize]
    not_follow_minstep = minstep[not_follow_idx]
    not_follow_eigenvals = eigenvalues[not_follow_idx]

    # Lambda_N via (n x n) eigenproblem
    tmp2 = torch.zeros(n, n, device=gradients.device, dtype=gradients.dtype)
    tmp2[:n - 1, :n - 1] = torch.diag(not_follow_eigenvals)
    tmp2[n - 1, :n - 1] = not_follow_minstep
    tmp2[:n - 1, n - 1] = not_follow_minstep
    lambda_n = torch.linalg.eigvalsh(tmp2)[0]

    # Step from followed mode
    denom_p = eigenvalues[mode_to_maximize] - lambda_p
    if abs(denom_p.item()) < 1e-20:
        denom_p = torch.tensor(1e-20, device=gradients.device, dtype=gradients.dtype)
    steps = (-minstep[mode_to_maximize] / denom_p) * eigenvectors[:, mode_to_maximize]

    # Steps from other modes
    for i in range(n):
        if i != mode_to_maximize:
            denom_n = eigenvalues[i] - lambda_n
            if abs(denom_n.item()) < 1e-20:
                denom_n = torch.tensor(1e-20, device=gradients.device, dtype=gradients.dtype)
            steps = steps - (minstep[i] / denom_n) * eigenvectors[:, i]

    if steps.isnan().any():
        raise RuntimeError("Step contains NaN")

    return steps, mode_to_maximize, followed_ev


def bofill_optimize(
    parameters: torch.Tensor,
    update_fn: Callable,
    settings: BofillSettings | None = None,
) -> tuple[torch.Tensor, int]:
    """Run Bofill optimization to find a saddle point.

    Args:
        parameters: Initial parameter vector of shape [n_params].
        update_fn: Callable ``(params, hessian_update_required) -> (value, gradients, hessian)``.
        settings: BofillSettings (defaults used if None).

    Returns:
        Tuple of (optimized_parameters, n_cycles).
    """
    if settings is None:
        settings = BofillSettings()

    n_params = parameters.shape[0]
    if n_params == 0:
        raise ValueError("Empty parameter vector")

    device = parameters.device
    dtype = parameters.dtype

    params = parameters.clone()

    # Initial evaluation with full Hessian
    hessian_update_required = True
    value, gradients, hessian_new = update_fn(params, hessian_update_required)
    hessian = hessian_new

    # Init convergence checker state (matches SCINE GradientBasedCheck)
    check_old_params = params.clone()
    check_old_value = value

    prev_eigenvector: torch.Tensor | None = None
    mode_to_follow = settings.mode_to_follow

    x_old = params.clone()
    g_old = gradients.clone()

    value_memory: deque[float] = deque(maxlen=settings.max_value_memory)

    cycle = 1  # matches SCINE _startCycle
    while cycle <= settings.max_iter:
        already_calculated = False
        cycle += 1

        try:
            steps, mode_to_follow, prev_eigenvector = _mode_maximization_with_hessian(
                gradients, hessian, mode_to_follow, prev_eigenvector,
            )
        except RuntimeError:
            if hessian_update_required:
                raise
            hessian_update_required = True
            value, gradients, hessian_new = update_fn(params, hessian_update_required)
            hessian = hessian_new
            steps, mode_to_follow, prev_eigenvector = _mode_maximization_with_hessian(
                gradients, hessian, mode_to_follow, prev_eigenvector,
            )
            already_calculated = True

        x_old = params.clone()
        max_val = steps.abs().max()
        if max_val > settings.trust_radius:
            steps = steps * (settings.trust_radius / max_val)
        params = params + steps

        g_old = gradients.clone()

        if not already_calculated:
            hessian_update_required = ((cycle - 1) % settings.hessian_update == 0)
            value, gradients, hessian_new = update_fn(params, hessian_update_required)
            if hessian_update_required:
                hessian = hessian_new

        # Convergence check (matches SCINE GradientBasedCheck)
        delta_param = params - check_old_params
        delta_v = value - check_old_value
        check_old_params = params.clone()
        check_old_value = value

        converged = 0
        if gradients.abs().max() < settings.grad_max_coeff:
            converged += 1
        if delta_param.abs().max() < settings.step_max_coeff:
            converged += 1
        if (gradients ** 2).sum().sqrt() / (n_params ** 0.5) < settings.grad_rms:
            converged += 1
        if (delta_param ** 2).sum().sqrt() / (n_params ** 0.5) < settings.step_rms:
            converged += 1

        stop = (cycle >= settings.max_iter) or (
            abs(delta_v) < settings.delta_value and converged >= settings.convergence_requirement
        )

        if stop:
            return params, cycle

        # Oscillation check (only if not converged)
        value_memory.append(value)
        if _is_oscillating(value_memory):
            x_old = params.clone()
            g_old = gradients.clone()
            params = params - steps / 2.0
            hessian_update_required = True
            value, gradients, hessian_new = update_fn(params, hessian_update_required)
            hessian = hessian_new
            cycle += 1

        # Bofill Hessian update (skip if full Hessian was just computed)
        if not hessian_update_required:
            dx = params - x_old
            dg = gradients - g_old
            dx_dot_dx = float((dx @ dx).item())
            if abs(dx_dot_dx) > 1e-20:
                tmp2 = dg - hessian @ dx
                tmp_dot_dx = float((tmp2 @ dx).item())

                den = float((tmp2 @ tmp2).item()) * float((dx @ dx).item())
                if abs(den) > 1e-20:
                    bofill_factor = (tmp_dot_dx ** 2) / den
                else:
                    bofill_factor = 1.0

                # PSB part
                hessian = hessian + (1 - bofill_factor) * (
                    (torch.outer(tmp2, dx) + torch.outer(dx, tmp2)) / dx_dot_dx
                )
                hessian = hessian - (1 - bofill_factor) * (
                    (tmp_dot_dx * torch.outer(dx, dx)) / (dx_dot_dx ** 2)
                )
                # SR1 part
                if abs(tmp_dot_dx) > 1e-20:
                    hessian = hessian + bofill_factor * (
                        torch.outer(tmp2, tmp2) / tmp_dot_dx
                    )

    return params, settings.max_iter


# ---------------------------------------------------------------------------
# SimState / ModelInterface wrapper
# ---------------------------------------------------------------------------

def bofill_ts_optimize(
    model: ModelInterface,
    state: SimState,
    settings: BofillSettings | None = None,
    hessian_delta: float = 0.01,
) -> tuple[SimState, int]:
    """Run Bofill saddle-point optimization on a single-system SimState.

    Wraps :func:`bofill_optimize` by constructing an ``update_fn`` that
    evaluates energy/forces via the ``ModelInterface`` and computes the
    semi-numeric Hessian (via :func:`compute_hessian`) when requested.

    Args:
        model: ModelInterface returning energy and forces.
        state: Single-system SimState (initial guess for the TS).
        settings: BofillSettings (defaults used if None).
        hessian_delta: Displacement for finite-difference Hessian.

    Returns:
        Tuple of (optimized SimState, number of cycles).
    """
    if state.n_systems != 1:
        raise ValueError(
            "bofill_ts_optimize expects a single-system SimState. "
            "Use batch_bofill_ts_optimize for multi-system batching."
        )

    n_atoms = state.n_atoms
    n_dof = 3 * n_atoms

    def _update_fn(
        params: torch.Tensor, hessian_update_required: bool,
    ) -> tuple[float, torch.Tensor, torch.Tensor | None]:
        pos = params.reshape(n_atoms, 3)
        s = SimState(
            positions=pos,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=state.atomic_numbers,
        )
        out = model(s)
        value = float(out["energy"].item())
        gradients = -out["forces"].detach().reshape(-1)

        hessian = None
        if hessian_update_required:
            hessian = compute_hessian(model, s, delta=hessian_delta)

        return value, gradients, hessian

    params = state.positions.detach().clone().reshape(-1)
    opt_params, n_cycles = bofill_optimize(params, _update_fn, settings)

    ts_state = SimState(
        positions=opt_params.reshape(n_atoms, 3),
        masses=state.masses.clone(),
        cell=state.cell.clone(),
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers.clone(),
    )
    return ts_state, n_cycles


def batch_bofill_ts_optimize(
    model: ModelInterface,
    state: SimState,
    settings_list: list[BofillSettings | None] | None = None,
    hessian_delta: float = 0.01,
) -> tuple[SimState, list[int]]:
    """Run Bofill saddle-point optimization on a batched SimState.

    Each system is optimized independently.  Energy/gradient evaluations
    are batched across systems within each cycle; Hessian evaluations are
    performed per-system (finite-difference perturbations cannot be shared
    across systems with different geometries).

    Args:
        model: ModelInterface returning energy and forces (batched).
        state: Batched SimState with ``n_systems`` systems.
        settings_list: Per-system BofillSettings.  If *None*, defaults
            are used for every system.
        hessian_delta: Displacement for finite-difference Hessian.

    Returns:
        Tuple of (optimized SimState, list of per-system cycle counts).
    """
    n_systems = state.n_systems
    if settings_list is None:
        settings_list = [None] * n_systems
    settings_list = [s or BofillSettings() for s in settings_list]

    system_idx = state.system_idx
    counts = torch.bincount(system_idx, minlength=n_systems)
    offsets = torch.zeros(n_systems, device=state.device, dtype=torch.long)
    offsets[1:] = counts[:-1].cumsum(0)

    result_positions: list[torch.Tensor] = []
    result_cycles: list[int] = []

    for s in range(n_systems):
        n_at = int(counts[s].item())
        off = int(offsets[s].item())
        single = SimState(
            positions=state.positions[off: off + n_at].clone(),
            masses=state.masses[off: off + n_at].clone(),
            cell=state.cell[s: s + 1].clone(),
            pbc=state.pbc,
            atomic_numbers=state.atomic_numbers[off: off + n_at].clone(),
        )
        ts, nc = bofill_ts_optimize(model, single, settings_list[s], hessian_delta)
        result_positions.append(ts.positions)
        result_cycles.append(nc)

    ts_state = SimState(
        positions=torch.cat(result_positions, dim=0),
        masses=state.masses.clone(),
        cell=state.cell.clone(),
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers.clone(),
        system_idx=system_idx.clone(),
    )
    return ts_state, result_cycles
