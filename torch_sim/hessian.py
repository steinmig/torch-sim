"""Semi-numeric Hessian computation for torch-sim models.

Computes the Hessian of the energy with respect to Cartesian positions
using central finite differences of the gradient (forces). Supports
batching of perturbations to reduce model-call overhead.
"""

from __future__ import annotations

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState


def compute_hessian(
    model: ModelInterface,
    state: SimState,
    delta: float = 0.01,
    batch_perturbations: bool = True,
) -> torch.Tensor:
    """Compute the Hessian via central finite differences of the gradient.

    For a single-system SimState with n_atoms atoms, displaces each of the
    3*n_atoms degrees of freedom by +/-delta, evaluates the model to get
    forces (= -gradient), and assembles the Hessian column by column.

    Args:
        model: A ModelInterface that returns forces.
        state: A single-system SimState.
        delta: Displacement magnitude in the same units as positions.
        batch_perturbations: If True, batch all +delta/-delta evaluations
            into a single model call (2 * 3*n_atoms systems).  Otherwise
            loop one DOF at a time (slower but uses less memory).

    Returns:
        Hessian matrix of shape [3*n_atoms, 3*n_atoms].
    """
    if state.n_systems != 1:
        raise ValueError("compute_hessian expects a single-system SimState")

    n_atoms = state.n_atoms
    n_dof = 3 * n_atoms
    positions = state.positions.detach().clone()
    device = state.device
    dtype = state.dtype

    if batch_perturbations:
        return _hessian_batched(model, state, positions, n_atoms, n_dof, delta, device, dtype)
    return _hessian_sequential(model, state, positions, n_atoms, n_dof, delta, device, dtype)


def _build_perturbed_state(
    state: SimState,
    all_positions: torch.Tensor,
    n_copies: int,
) -> SimState:
    """Build a batched SimState with n_copies copies of the original system."""
    device = state.device
    dtype = state.dtype
    n_atoms_single = state.n_atoms
    total_atoms = n_copies * n_atoms_single

    masses = state.masses.repeat(n_copies)
    atomic_numbers = state.atomic_numbers.repeat(n_copies)
    cell = state.cell.repeat(n_copies, 1, 1)
    pbc = state.pbc

    system_idx = torch.arange(n_copies, device=device, dtype=torch.int64).repeat_interleave(n_atoms_single)

    return SimState(
        positions=all_positions.reshape(total_atoms, 3),
        masses=masses,
        cell=cell,
        pbc=pbc,
        atomic_numbers=atomic_numbers,
        system_idx=system_idx,
    )


def _hessian_batched(
    model: ModelInterface,
    state: SimState,
    positions: torch.Tensor,
    n_atoms: int,
    n_dof: int,
    delta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute Hessian by batching all perturbations into two model calls."""
    pos_flat = positions.reshape(-1)

    plus_positions = pos_flat.unsqueeze(0).expand(n_dof, -1).clone()
    minus_positions = pos_flat.unsqueeze(0).expand(n_dof, -1).clone()

    idx = torch.arange(n_dof, device=device)
    plus_positions[idx, idx] += delta
    minus_positions[idx, idx] -= delta

    plus_state = _build_perturbed_state(
        state, plus_positions.reshape(n_dof, n_atoms, 3), n_dof
    )
    minus_state = _build_perturbed_state(
        state, minus_positions.reshape(n_dof, n_atoms, 3), n_dof
    )

    plus_out = model(plus_state)
    minus_out = model(minus_state)

    plus_forces = plus_out["forces"]
    minus_forces = minus_out["forces"]

    plus_grad = -plus_forces.reshape(n_dof, n_dof)
    minus_grad = -minus_forces.reshape(n_dof, n_dof)

    hessian = (plus_grad - minus_grad) / (2.0 * delta)
    hessian = (hessian + hessian.T) / 2.0

    return hessian


def _hessian_sequential(
    model: ModelInterface,
    state: SimState,
    positions: torch.Tensor,
    n_atoms: int,
    n_dof: int,
    delta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute Hessian one DOF at a time (low memory)."""
    pos_flat = positions.reshape(-1)
    hessian = torch.zeros((n_dof, n_dof), device=device, dtype=dtype)

    for i in range(n_dof):
        plus_pos = pos_flat.clone()
        minus_pos = pos_flat.clone()
        plus_pos[i] += delta
        minus_pos[i] -= delta

        plus_state = SimState(
            positions=plus_pos.reshape(n_atoms, 3),
            masses=state.masses.clone(),
            cell=state.cell.clone(),
            pbc=state.pbc,
            atomic_numbers=state.atomic_numbers.clone(),
        )
        minus_state = SimState(
            positions=minus_pos.reshape(n_atoms, 3),
            masses=state.masses.clone(),
            cell=state.cell.clone(),
            pbc=state.pbc,
            atomic_numbers=state.atomic_numbers.clone(),
        )

        plus_out = model(plus_state)
        minus_out = model(minus_state)

        plus_grad = -plus_out["forces"].reshape(-1)
        minus_grad = -minus_out["forces"].reshape(-1)

        hessian[:, i] = (plus_grad - minus_grad) / (2.0 * delta)

    hessian = (hessian + hessian.T) / 2.0
    return hessian
