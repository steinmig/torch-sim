"""NT2 (Newton Trajectory 2) optimizer for transition state search.

Reimplements the SCINE NtOptimizer2 algorithm in the torch-sim framework.
The optimizer pushes atom pairs together (association) or apart (dissociation)
while relaxing the rest of the structure, recording the energy trajectory
and extracting a TS guess from the highest-energy point.

All default parameter values are in Angstrom / eV units, converted from
SCINE's Bohr / Hartree defaults.

Supports batched execution: multiple independent NT2 runs share a single
batched model call per cycle, while per-system gradient manipulation and
convergence tracking run in a lightweight Python loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from torch_sim.bonds import detect_bonds
from torch_sim.state import SimState

if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface

# SCINE uses Bohr / Hartree.  torch-sim uses Angstrom / eV.
_BOHR_TO_ANG = 0.529177
_HARTREE_TO_EV = 27.2114
_HA_BOHR_TO_EV_ANG = _HARTREE_TO_EV / _BOHR_TO_ANG  # ~51.422 eV/A per Ha/Bohr
_BOHR2_PER_HARTREE = _BOHR_TO_ANG ** 2 / _HARTREE_TO_EV  # ~0.01029


@dataclass
class NT2Settings:
    """Settings for the NT2 optimizer (Angstrom / eV units)."""
    total_force_norm: float = 0.1 * _HA_BOHR_TO_EV_ANG  # ~5.14 eV/A
    sd_factor: float = 1.0 * _BOHR2_PER_HARTREE  # ~0.0103
    max_iter: int = 500
    attractive_distance_stop: float = 0.9
    attractive_bond_order_stop: float = 0.75
    repulsive_bond_order_stop: float = 0.15
    use_micro_cycles: bool = True
    number_of_micro_cycles: int = 10
    filter_passes: int = 10
    extraction_criterion: str = "lastBeforeTarget"
    extra_macrocycles_after_bond_criteria: int = 10
    dissociation_scale: float = 0.8 * _BOHR_TO_ANG  # ~0.42 A
    min_association_scale: float = 0.1 * _BOHR_TO_ANG  # ~0.053 A
    fixed_atoms: list[int] = field(default_factory=list)
    cov_radii: dict[int, float] | None = None
    bond_det_cov_radii: dict[int, float] | None = None
    bond_tolerance: float | None = None


# ---------------------------------------------------------------------------
# Covalent radii helpers
# ---------------------------------------------------------------------------

_COVALENT_RADII_ANGSTROM: dict[int, float] | None = None


def _get_cov_radii() -> dict[int, float]:
    global _COVALENT_RADII_ANGSTROM
    if _COVALENT_RADII_ANGSTROM is None:
        from ase.data import covalent_radii
        _COVALENT_RADII_ANGSTROM = {z: float(covalent_radii[z]) for z in range(len(covalent_radii))}
    return _COVALENT_RADII_ANGSTROM


def _smallest_cov_radius(
    atomic_numbers: torch.Tensor,
    indices: list[int],
    cov_radii: dict[int, float] | None = None,
) -> float:
    table = cov_radii if cov_radii is not None else _get_cov_radii()
    return min(table.get(int(atomic_numbers[i].item()), 1.5) for i in indices)


# ---------------------------------------------------------------------------
# Geometry / graph helpers
# ---------------------------------------------------------------------------

def _center_to_center(positions: torch.Tensor, left: list[int], right: list[int]) -> torch.Tensor:
    """Vector from center of right to center of left (matches SCINE convention)."""
    left_center = positions[left].mean(dim=0)
    right_center = positions[right].mean(dim=0)
    return left_center - right_center


def _connected_nuclei(indices: list[int], bond_matrix: torch.Tensor) -> list[list[int]]:
    """Group indices into connected components based on bond_matrix."""
    remaining = set(indices)
    components: list[list[int]] = []
    while remaining:
        seed = next(iter(remaining))
        component = []
        stack = [seed]
        while stack:
            node = stack.pop()
            if node in remaining:
                remaining.discard(node)
                component.append(node)
                for other in list(remaining):
                    if bond_matrix[node, other] > 0.5:
                        stack.append(other)
        components.append(sorted(component))
    return components


def _infer_reactions(
    association_list: list[int],
    dissociation_list: list[int],
    bond_matrix: torch.Tensor,
) -> tuple[list[tuple[list[int], list[int]]], list[tuple[list[int], list[int]]]]:
    """Build reaction mappings from flat index lists, handling eta-bonds."""
    maps = []
    for rxn_list in (association_list, dissociation_list):
        pairs = [(rxn_list[2 * i], rxn_list[2 * i + 1]) for i in range(len(rxn_list) // 2)]

        right_corrected: list[tuple[int, list[int]]] = []
        seen_left: set[int] = set()
        for i, (l, r) in enumerate(pairs):
            if l in seen_left:
                continue
            seen_left.add(l)
            values = [r]
            for j in range(i + 1, len(pairs)):
                if pairs[j][0] == l:
                    values.append(pairs[j][1])
            if len(values) > 1:
                molecules = _connected_nuclei(values, bond_matrix)
                for mol in molecules:
                    right_corrected.append((l, mol))
            else:
                right_corrected.append((l, values))

        all_corrected: list[tuple[list[int], list[int]]] = []
        seen_right: list[list[int]] = []
        for i, (left_idx, right_list) in enumerate(right_corrected):
            if right_list in seen_right:
                continue
            seen_right.append(right_list)
            left_values = [left_idx]
            for j in range(i + 1, len(right_corrected)):
                if right_corrected[j][1] == right_list:
                    left_values.append(right_corrected[j][0])
            if len(left_values) > 1:
                molecules = _connected_nuclei(left_values, bond_matrix)
                for mol in molecules:
                    all_corrected.append((mol, right_list))
            else:
                all_corrected.append((left_values, right_list))

        maps.append(all_corrected)
    return maps[0], maps[1]


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

def _build_reactive_atoms(association_list: list[int], dissociation_list: list[int]) -> list[int]:
    combined = sorted(set(association_list + dissociation_list))
    return combined


def _build_constraints_map(
    n_atoms: int,
    reactive_atoms: list[int],
    association_list: list[int],
    dissociation_list: list[int],
) -> list[list[int]]:
    """For each atom, list the pairs of indices it participates in."""
    constraints_map: list[list[int]] = [[] for _ in range(n_atoms)]
    for idx in reactive_atoms:
        matches: list[int] = []
        for k in range(len(association_list) // 2):
            l, r = association_list[2 * k], association_list[2 * k + 1]
            if idx == l or idx == r:
                matches.extend([l, r])
        for k in range(len(dissociation_list) // 2):
            l, r = dissociation_list[2 * k], dissociation_list[2 * k + 1]
            if idx == l or idx == r:
                matches.extend([l, r])
        constraints_map[idx] = matches
    return constraints_map


# ---------------------------------------------------------------------------
# Gradient manipulation (per-system, on a single system's tensors)
# ---------------------------------------------------------------------------

def _eliminate_reactive_gradients(
    positions: torch.Tensor,
    gradients: torch.Tensor,
    reactive_atoms: list[int],
    constraints_map: list[list[int]],
) -> torch.Tensor:
    """Project out gradient components along reaction-coordinate directions."""
    grad = gradients.clone()
    for atom_idx in reactive_atoms:
        pairs = constraints_map[atom_idx]
        n_pairs = len(pairs) // 2
        if n_pairs == 0:
            continue

        axes = []
        for j in range(n_pairs):
            l, r = pairs[2 * j], pairs[2 * j + 1]
            axis = positions[l] - positions[r]
            norm = axis.norm()
            if norm > 1e-12:
                axes.append(axis / norm)

        if not axes:
            continue

        axes_mat = torch.stack(axes, dim=0)
        ata = axes_mat @ axes_mat.T
        eigenvalues, eigenvectors = torch.linalg.eigh(ata)
        ortho_axes = (axes_mat.T @ eigenvectors).T

        rank = int((eigenvalues > 1e-6).sum().item())

        if rank > 2:
            grad[atom_idx] = 0.0
        elif rank == 2:
            a0 = ortho_axes[-1]
            a0 = a0 / a0.norm()
            a1 = ortho_axes[-2]
            a1 = a1 / a1.norm()
            proj = torch.cross(a0, a1)
            proj = proj / proj.norm()
            grad[atom_idx] = (grad[atom_idx] @ proj) * proj
        elif rank == 1:
            a = ortho_axes[-1]
            a = a / a.norm()
            grad[atom_idx] = grad[atom_idx] - (grad[atom_idx] @ a) * a

    return grad


def _update_gradients(
    positions: torch.Tensor,
    gradients: torch.Tensor,
    atomic_numbers: torch.Tensor,
    bond_matrix: torch.Tensor,
    association_list: list[int],
    dissociation_list: list[int],
    reactive_atoms: list[int],
    constraints_map: list[list[int]],
    settings: NT2Settings,
    cycle: int,
    add_force: bool,
    in_extra_macrocycles: bool,
    first_coord_reached: int,
) -> tuple[torch.Tensor, int]:
    """Apply reaction-coordinate forces and project reactive-atom gradients."""
    grad = _eliminate_reactive_gradients(positions, gradients, reactive_atoms, constraints_map)

    associations, dissociations = _infer_reactions(association_list, dissociation_list, bond_matrix)

    rxn_coord = torch.zeros_like(grad)
    max_scale = 0.0

    custom_radii = settings.cov_radii

    for left, right in associations:
        r12cov = (_smallest_cov_radius(atomic_numbers, left, custom_radii)
                  + _smallest_cov_radius(atomic_numbers, right, custom_radii))
        c2c = _center_to_center(positions, left, right)
        dist = c2c.norm().item()

        bo = sum(float(bond_matrix[l, r].item()) for l in left for r in right)

        if not in_extra_macrocycles and (
            bo > 1.1 * settings.attractive_bond_order_stop
            or dist < 0.9 * settings.attractive_distance_stop * r12cov
        ):
            if first_coord_reached == -1:
                first_coord_reached = cycle
            continue

        scale = dist - r12cov
        if scale < 0:
            scale = settings.min_association_scale
        max_scale = max(max_scale, scale)
        direction = c2c * (scale / max(dist, 1e-12))
        for l in left:
            for r in right:
                rxn_coord[l] += direction
                rxn_coord[r] -= direction

    for left, right in dissociations:
        bo = sum(float(bond_matrix[l, r].item()) for l in left for r in right)
        c2c = _center_to_center(positions, left, right)
        dist = c2c.norm().item()

        if not in_extra_macrocycles and bo < 0.7 * settings.repulsive_bond_order_stop:
            if first_coord_reached == -1:
                first_coord_reached = cycle
            continue

        scale = settings.dissociation_scale
        max_scale = max(max_scale, scale)
        direction = c2c * (scale / max(dist, 1e-12))
        for l in left:
            for r in right:
                rxn_coord[l] -= direction
                rxn_coord[r] += direction

    if max_scale > 0:
        rxn_coord *= 0.5 * (settings.total_force_norm / max_scale)

    if add_force:
        grad = grad + rxn_coord

    if settings.fixed_atoms:
        for a in settings.fixed_atoms:
            grad[a] = 0.0

    return grad, first_coord_reached


def _converged(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    bond_matrix: torch.Tensor,
    association_list: list[int],
    dissociation_list: list[int],
    settings: NT2Settings,
) -> bool:
    """Check if all reaction coordinates have met their criteria."""
    associations, dissociations = _infer_reactions(association_list, dissociation_list, bond_matrix)

    custom_radii = settings.cov_radii

    for left, right in associations:
        r12cov = (_smallest_cov_radius(atomic_numbers, left, custom_radii)
                  + _smallest_cov_radius(atomic_numbers, right, custom_radii))
        c2c = _center_to_center(positions, left, right)
        dist = c2c.norm().item()
        bo = sum(float(bond_matrix[l, r].item()) for l in left for r in right)
        if bo < settings.attractive_bond_order_stop and dist > settings.attractive_distance_stop * r12cov:
            return False

    for left, right in dissociations:
        bo = sum(float(bond_matrix[l, r].item()) for l in left for r in right)
        if bo > settings.repulsive_bond_order_stop:
            return False

    return True


# ---------------------------------------------------------------------------
# TS guess extraction
# ---------------------------------------------------------------------------

def _extract_ts_guess(
    values: list[float],
    trajectory: list[torch.Tensor],
    filter_passes: int,
    extraction_criterion: str,
    first_coord_reached: int,
) -> torch.Tensor:
    """Extract TS guess from the energy trajectory using Savitzky-Golay filtering."""
    n = len(values)
    if n < 3:
        return trajectory[-1]

    filtered = list(values)
    filtered_grad = [0.0] * n

    for _ in range(filter_passes):
        tmp = [filtered[0], filtered[0]] + filtered + [filtered[-1], filtered[-1]]
        for j in range(2, n + 2):
            a0 = (-3.0 * tmp[j - 2] + 12.0 * tmp[j - 1] + 17.0 * tmp[j] + 12.0 * tmp[j + 1] - 3.0 * tmp[j + 2]) / 35.0
            a1 = (tmp[j - 2] - 8.0 * tmp[j - 1] + 8.0 * tmp[j + 1] - tmp[j + 2]) / 12.0
            filtered[j - 2] = a0
            filtered_grad[j - 2] = a1

    maxima: list[int] = []
    for i in range(n - 2, 0, -1):
        if filtered_grad[i] >= 0.0 and filtered_grad[i + 1] < 0.0:
            idx = i if abs(filtered_grad[i]) < abs(filtered_grad[i + 1]) else i + 1
            maxima.append(idx)

    if not maxima:
        return trajectory[values.index(max(values))]

    if extraction_criterion == "first":
        return trajectory[maxima[-1]]

    if extraction_criterion == "highest" or first_coord_reached == -1:
        best_idx = max(maxima, key=lambda i: values[i])
        return trajectory[best_idx]

    for m in maxima:
        if m < first_coord_reached:
            return trajectory[m]

    return trajectory[maxima[-1]]


# ---------------------------------------------------------------------------
# Per-system internal state tracked during the optimization loop
# ---------------------------------------------------------------------------

@dataclass
class _SystemTracker:
    """Mutable state for one system in a batched NT2 run."""
    association_list: list[int]
    dissociation_list: list[int]
    settings: NT2Settings
    reactive_atoms: list[int]
    constraints_map: list[list[int]]
    n_atoms: int
    atom_offset: int
    atomic_numbers: torch.Tensor
    values: list[float] = field(default_factory=list)
    trajectory: list[torch.Tensor] = field(default_factory=list)
    first_coord_reached: int = -1
    extra_macrocycles_remaining: int = -1
    finished: bool = False


# ---------------------------------------------------------------------------
# BFGS micro-cycle optimizer  (matches SCINE Utils::Bfgs with trust radius)
# ---------------------------------------------------------------------------

def _is_oscillating(value_memory: deque) -> bool:
    """Check if the energy history shows an oscillating pattern (alternating signs)."""
    n = len(value_memory)
    if n < 3:
        return False
    if abs(value_memory[0] - value_memory[1]) < 1e-12:
        return False
    prev_positive = (value_memory[0] - value_memory[1]) > 0.0
    for i in range(2, n):
        this_positive = (value_memory[i - 1] - value_memory[i]) > 0.0
        if prev_positive == this_positive:
            return False
        prev_positive = this_positive
    return True


def _gdiis_update(
    inv_h: torch.Tensor,
    params: torch.Tensor,
    grad: torch.Tensor,
    x_store: torch.Tensor,
    g_store: torch.Tensor,
    gdiis_cycle: int,
    max_store: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """GDIIS extrapolation (matches SCINE Utils::Gdiis::update).

    Returns (params, grad, gdiis_cycle) – params/grad may be modified in-place.
    If the extrapolation would raise the energy, the history is flushed.
    """
    if max_store < 2:
        return params, grad, gdiis_cycle

    col = gdiis_cycle % max_store
    gdiis_cycle += 1
    x_store[:, col] = params
    g_store[:, col] = grad

    n = min(gdiis_cycle, max_store)
    if gdiis_cycle < 2:
        return params, grad, gdiis_cycle

    dx = torch.zeros_like(g_store[:, :n])
    for i in range(n):
        dx[:, i] = -(inv_h @ g_store[:, i])

    B = torch.zeros(n + 1, n + 1, dtype=params.dtype, device=params.device)
    B[:n, :n] = dx[:, :n].T @ dx[:, :n]
    B[n, :n] = 1.0
    B[:n, n] = 1.0

    rhs = torch.zeros(n + 1, dtype=params.dtype, device=params.device)
    rhs[n] = 1.0
    try:
        coeffs = torch.linalg.solve(B, rhs)
    except torch.linalg.LinAlgError:
        return params, grad, 0  # flush on singular B
    c_sum = coeffs[:n].sum().item()
    if abs(c_sum) < 1e-12:
        return params, grad, 0  # flush
    coeffs /= c_sum

    tmp_grad = (g_store[:, :n] @ coeffs[:n])
    tmp_param = (x_store[:, :n] @ coeffs[:n])

    dx_diis = tmp_param - params
    dx_norm = dx_diis.norm().item()
    grad_norm = tmp_grad.norm().item()
    if dx_norm > 1e-12 and grad_norm > 1e-12:
        exp_change = float(tmp_grad.dot(dx_diis).item()) / (grad_norm * dx_norm)
        if exp_change > 1e-4:
            return params, grad, 0  # flush

    params = tmp_param
    grad = tmp_grad
    return params, grad, gdiis_cycle


def _batched_bfgs_micro_cycles(
    model: "ModelInterface",
    positions: torch.Tensor,
    state: SimState,
    trackers: list[_SystemTracker],
) -> None:
    """Run BFGS+GDIIS micro-cycle relaxation with batched model calls.

    All active systems are evaluated together in a single batched model call
    per micro-iteration, then each system's BFGS state is updated
    independently.  This avoids the O(N*M) sequential model calls of the
    per-system approach while preserving identical per-system BFGS logic
    (trust-radius clamping, Powell damping, GDIIS, oscillation correction).
    """
    active: list[tuple[int, _SystemTracker]] = [
        (s, tr) for s, tr in enumerate(trackers)
        if not tr.finished and tr.settings.use_micro_cycles and tr.n_atoms > 2
    ]
    if not active:
        return

    max_micro = max(tr.settings.number_of_micro_cycles for _, tr in active)
    dev, dt = positions.device, positions.dtype
    trust_radius = 0.2
    gdiis_max_store = 5

    # ---- per-system BFGS state ----
    bs: dict[int, dict] = {}
    for s, tr in active:
        n_p = tr.n_atoms * 3
        bs[s] = {
            "inv_h": 0.5 * torch.eye(n_p, dtype=dt, device=dev),
            "x_store": torch.zeros(n_p, gdiis_max_store, dtype=dt, device=dev),
            "g_store": torch.zeros(n_p, gdiis_max_store, dtype=dt, device=dev),
            "gdiis_cycle": 0,
            "value_memory": deque(maxlen=10),
            "bfgs_cycle": 0,
            "params": None,
            "params_old": None,
            "grad_old": None,
            "old_value": 0.0,
        }

    # ---- pre-compute static parts of the batched micro state ----
    mass_l, z_l, cell_l, sidx_l = [], [], [], []
    for idx, (s, tr) in enumerate(active):
        off, n_at = tr.atom_offset, tr.n_atoms
        mass_l.append(state.masses[off: off + n_at])
        z_l.append(tr.atomic_numbers)
        cell_l.append(state.cell[s: s + 1])
        sidx_l.append(torch.full((n_at,), idx, device=dev, dtype=torch.long))
    static_masses = torch.cat(mass_l)
    static_z = torch.cat(z_l)
    static_cell = torch.cat(cell_l)
    static_sidx = torch.cat(sidx_l)

    def _eval_all() -> dict[int, tuple[float, torch.Tensor]]:
        pos_l = [positions[tr.atom_offset: tr.atom_offset + tr.n_atoms].clone()
                 for _, tr in active]
        batch = SimState(
            positions=torch.cat(pos_l),
            masses=static_masses,
            cell=static_cell,
            pbc=state.pbc,
            atomic_numbers=static_z,
            system_idx=static_sidx,
        )
        out = model(batch)

        results: dict[int, tuple[float, torch.Tensor]] = {}
        a_off = 0
        for idx, (s, tr) in enumerate(active):
            n_at = tr.n_atoms
            val = float(out["energy"][idx].item())
            grad = -out["forces"][a_off: a_off + n_at].detach()
            grad = _eliminate_reactive_gradients(
                positions[tr.atom_offset: tr.atom_offset + n_at], grad,
                tr.reactive_atoms, tr.constraints_map,
            )
            if tr.settings.fixed_atoms:
                for a in tr.settings.fixed_atoms:
                    grad[a] = 0.0
            results[s] = (val, grad)
            a_off += n_at
        return results

    def _reset_inv_h(st: dict, n_p: int, dg: torch.Tensor, dx_t_dg: float):
        dg_t_dg = float(dg.dot(dg).item())
        if dg_t_dg > 1e-9:
            st["inv_h"] = torch.eye(n_p, dtype=dt, device=dev) * (dx_t_dg / dg_t_dg)
        else:
            st["inv_h"] = 0.5 * torch.eye(n_p, dtype=dt, device=dev)

    # ---- initial evaluation (batched) ----
    results = _eval_all()
    for s, tr in active:
        st = bs[s]
        off, n_at = tr.atom_offset, tr.n_atoms
        val, grad_mat = results[s]
        grad = grad_mat.reshape(-1)
        params = positions[off: off + n_at].clone().reshape(-1)

        st["params"] = params
        st["params_old"] = params.clone()
        st["grad_old"] = grad.clone()
        st["old_value"] = val

        step = -(st["inv_h"] @ grad)
        if trust_radius > 0:
            mx = step.abs().max().item()
            if mx > trust_radius:
                step = step * (trust_radius / mx)
        st["params"] = params + step
        positions[off: off + n_at] = st["params"].reshape(n_at, 3)
        st["bfgs_cycle"] = 1

    # ---- main micro-cycle loop (batched eval, per-system BFGS update) ----
    for _mc in range(max_micro):
        results = _eval_all()

        for s, tr in active:
            if _mc >= tr.settings.number_of_micro_cycles:
                continue

            st = bs[s]
            off, n_at = tr.atom_offset, tr.n_atoms
            n_p = n_at * 3
            st["bfgs_cycle"] += 1

            val, grad_mat = results[s]
            grad = grad_mat.reshape(-1)
            params = st["params"]
            params_old = st["params_old"]
            grad_old = st["grad_old"]
            inv_h = st["inv_h"]

            # Convergence check (rarely triggers; kept for correctness)
            if st["bfgs_cycle"] > 2:
                p_step = params - params_old
                n_met = sum([
                    float(p_step.abs().max().item()) < 1e-4,
                    float((p_step ** 2).mean().sqrt().item()) < 5e-4,
                    float(grad.abs().max().item()) < 5e-5,
                    float((grad ** 2).mean().sqrt().item()) < 1e-5,
                ])
                if abs(val - st["old_value"]) < 1e-7 and n_met >= 3:
                    continue
            st["old_value"] = val

            # BFGS inverse-Hessian update
            dx = params - params_old
            dg = grad - grad_old
            dx_t_dg = float(dx.dot(dg).item())
            dg_t_inv_h_dg = float((dg @ inv_h @ dg).item())

            if st["bfgs_cycle"] == 2:
                dg_t_dg = float(dg.dot(dg).item())
                if dg_t_dg > 1e-12:
                    inv_h.diagonal().mul_(2.0 * dx_t_dg / dg_t_dg)

            sigma2, sigma3 = 0.9, 9.0
            delta = 1.0
            if abs(dx_t_dg) < abs((1.0 - sigma2) * dg_t_inv_h_dg):
                delta = sigma2 * dg_t_inv_h_dg / (dg_t_inv_h_dg - dx_t_dg)
            elif abs(dx_t_dg) > abs((1.0 + sigma3) * dg_t_inv_h_dg):
                delta = -sigma3 * dg_t_inv_h_dg / (dg_t_inv_h_dg - dx_t_dg)
            if abs(delta - 1.0) > 1e-12:
                dx = delta * dx + (1.0 - delta) * (inv_h @ dg)
                dx_t_dg = float(dx.dot(dg).item())
            if abs(dx_t_dg) < 1e-9:
                dx_t_dg = -1e-9 if dx_t_dg < 0.0 else 1e-9

            if dx_t_dg > 0.0:
                alpha = (dx_t_dg + float((dg @ inv_h @ dg).item())) / (dx_t_dg ** 2)
                beta = 1.0 / dx_t_dg
                inv_h = inv_h + alpha * torch.outer(dx, dx) - beta * (
                    inv_h @ torch.outer(dg, dx) + torch.outer(dx, dg) @ inv_h
                )
            else:
                inv_h = 0.5 * torch.eye(n_p, dtype=dt, device=dev)
                st["gdiis_cycle"] = 0

            st["inv_h"] = inv_h
            st["params_old"] = params.clone()
            st["grad_old"] = grad.clone()

            params, grad, st["gdiis_cycle"] = _gdiis_update(
                inv_h, params, grad,
                st["x_store"], st["g_store"], st["gdiis_cycle"], gdiis_max_store,
            )

            step = -(inv_h @ grad)

            if trust_radius > 0:
                projected = params + step - st["params_old"]
                mx = projected.abs().max().item()
                if mx > trust_radius:
                    step = projected * (trust_radius / mx)
                    params = st["params_old"].clone()
                    _reset_inv_h(st, n_p, dg, dx_t_dg)
                    st["gdiis_cycle"] = 0

            params = params + step
            positions[off: off + n_at] = params.reshape(n_at, 3)
            st["params"] = params

            st["value_memory"].append(val)
            if _is_oscillating(st["value_memory"]):
                params = params - step / 2.0
                positions[off: off + n_at] = params.reshape(n_at, 3)
                st["params"] = params
                _reset_inv_h(st, n_p, dg, dx_t_dg)
                st["value_memory"].clear()
                st["gdiis_cycle"] = 0


# ---------------------------------------------------------------------------
# Public API: single-system (backward-compat) and batched
# ---------------------------------------------------------------------------

def nt2_optimize(
    model: "ModelInterface",
    state: SimState,
    association_list: list[int],
    dissociation_list: list[int],
    settings: NT2Settings | None = None,
) -> tuple[SimState, list[torch.Tensor], list[float]]:
    """Run a *single-system* NT2 optimization.

    Convenience wrapper around :func:`batch_nt2_optimize` for the common
    single-system case.

    Args:
        model: ModelInterface for energy/forces.
        state: Initial single-system SimState.
        association_list: Flat list of atom-index pairs to associate.
        dissociation_list: Flat list of atom-index pairs to dissociate.
        settings: NT2Settings (defaults used if None).

    Returns:
        Tuple of (ts_guess_state, trajectory_positions, energy_values).
    """
    if state.n_systems != 1:
        raise ValueError(
            "nt2_optimize expects a single-system SimState. "
            "Use batch_nt2_optimize for multi-system batching."
        )

    results = batch_nt2_optimize(
        model=model,
        state=state,
        association_lists=[association_list],
        dissociation_lists=[dissociation_list],
        settings_list=[settings] if settings else None,
    )
    ts_state, trajs, vals = results
    # For single system the returned state is already single-system;
    # return the first (only) trajectory and values.
    return ts_state, trajs[0], vals[0]


def batch_nt2_optimize(
    model: "ModelInterface",
    state: SimState,
    association_lists: list[list[int]],
    dissociation_lists: list[list[int]],
    settings_list: list[NT2Settings | None] | None = None,
) -> tuple[SimState, list[list[torch.Tensor]], list[list[float]]]:
    """Run NT2 optimization on a (possibly batched) SimState.

    Each system in *state* is an independent NT2 run.  The model is
    called once per macro-cycle on *all active systems* in a single
    batched forward pass, giving the GPU-utilisation benefit.  The
    cheap per-system gradient manipulation and convergence logic run
    in a lightweight Python loop.

    Args:
        model: ModelInterface for energy/forces (batched).
        state: SimState with ``n_systems`` systems.
        association_lists: Per-system association index pairs.  Length
            must equal ``state.n_systems``.
        dissociation_lists: Per-system dissociation index pairs.
        settings_list: Per-system settings.  If *None*, defaults are
            used for every system.

    Returns:
        Tuple of:
        - ts_guess_state: SimState with one system per input system
          (positions replaced by extracted TS guess).
        - trajectories: ``list[list[Tensor]]`` -- per-system list of
          position snapshots (each ``[n_atoms_i, 3]``).
        - energies: ``list[list[float]]`` -- per-system energy traces.
    """
    n_systems = state.n_systems
    if len(association_lists) != n_systems:
        raise ValueError(
            f"Got {len(association_lists)} association lists but state has {n_systems} systems"
        )
    if len(dissociation_lists) != n_systems:
        raise ValueError(
            f"Got {len(dissociation_lists)} dissociation lists but state has {n_systems} systems"
        )
    if settings_list is None:
        settings_list = [None] * n_systems
    settings_list = [s or NT2Settings() for s in settings_list]

    for s_idx in range(n_systems):
        if not association_lists[s_idx] and not dissociation_lists[s_idx]:
            raise ValueError(
                f"System {s_idx}: at least one of association/dissociation list must be non-empty"
            )

    # Compute per-system atom counts and offsets (systems are contiguous in SimState)
    system_idx = state.system_idx
    counts = torch.bincount(system_idx, minlength=n_systems)
    offsets = torch.zeros(n_systems, device=state.device, dtype=torch.long)
    offsets[1:] = counts[:-1].cumsum(0)

    # Build per-system trackers
    trackers: list[_SystemTracker] = []
    for s in range(n_systems):
        n_at = int(counts[s].item())
        off = int(offsets[s].item())
        a_list = association_lists[s]
        d_list = dissociation_lists[s]
        sett = settings_list[s]
        reactive = _build_reactive_atoms(a_list, d_list)
        cmap = _build_constraints_map(n_at, reactive, a_list, d_list)
        at_nums = state.atomic_numbers[off: off + n_at]
        trackers.append(_SystemTracker(
            association_list=a_list,
            dissociation_list=d_list,
            settings=sett,
            reactive_atoms=reactive,
            constraints_map=cmap,
            n_atoms=n_at,
            atom_offset=off,
            atomic_numbers=at_nums,
        ))

    # Working copy of all positions  [n_total_atoms, 3]
    positions = state.positions.detach().clone()
    max_iter = max(s.settings.max_iter for s in trackers)

    # ---- main loop (matches SCINE NtOptimizer2::optimize ordering) ----
    for cycle in range(1, max_iter + 1):
        if all(t.finished for t in trackers):
            break

        # 1) BFGS micro cycles FIRST (cycle > 1), before the macro evaluation.
        #    All active systems are evaluated together in a single batched
        #    model call per micro-iteration.
        if cycle > 1:
            _batched_bfgs_micro_cycles(model, positions, state, trackers)

        # 2) Build batched state and call model ONCE
        current_state = SimState(
            positions=positions.clone(),
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=state.atomic_numbers,
            system_idx=system_idx,
        )
        output = model(current_state)
        all_forces = output["forces"].detach()   # [n_total_atoms, 3]
        all_energies = output["energy"].detach()  # [n_systems]

        # 3) Per-system gradient manipulation and convergence (cheap)
        for s, tr in enumerate(trackers):
            if tr.finished:
                continue
            if cycle > tr.settings.max_iter:
                tr.finished = True
                continue

            off = tr.atom_offset
            n_at = tr.n_atoms
            pos_s = positions[off: off + n_at]
            forces_s = all_forces[off: off + n_at]
            energy_s = float(all_energies[s].item())
            gradients_s = -forces_s

            in_extra = tr.extra_macrocycles_remaining >= 0
            bond_kwargs: dict = {}
            bd_radii = tr.settings.bond_det_cov_radii or tr.settings.cov_radii
            if bd_radii is not None:
                bond_kwargs["cov_radii"] = bd_radii
            if tr.settings.bond_tolerance is not None:
                bond_kwargs["tolerance"] = tr.settings.bond_tolerance
            bond_mat = detect_bonds(tr.atomic_numbers, pos_s, **bond_kwargs)

            grad_mod, tr.first_coord_reached = _update_gradients(
                pos_s, gradients_s, tr.atomic_numbers, bond_mat,
                tr.association_list, tr.dissociation_list,
                tr.reactive_atoms, tr.constraints_map, tr.settings, cycle,
                add_force=True, in_extra_macrocycles=in_extra,
                first_coord_reached=tr.first_coord_reached,
            )

            tr.values.append(energy_s)
            tr.trajectory.append(pos_s.clone())

            if _converged(pos_s, tr.atomic_numbers, bond_mat,
                          tr.association_list, tr.dissociation_list, tr.settings):
                if tr.extra_macrocycles_remaining == -1:
                    tr.extra_macrocycles_remaining = tr.settings.extra_macrocycles_after_bond_criteria
                if tr.extra_macrocycles_remaining == 0:
                    tr.finished = True
                    continue
                tr.extra_macrocycles_remaining -= 1
            else:
                tr.extra_macrocycles_remaining = -1

            # Macro SD step: positions -= factor * modified_gradients
            new_pos = pos_s - tr.settings.sd_factor * grad_mod
            if tr.settings.fixed_atoms:
                orig = tr.trajectory[0]
                for a in tr.settings.fixed_atoms:
                    new_pos[a] = orig[a]
            positions[off: off + n_at] = new_pos

    # ---- extract TS guesses and build result state ----
    ts_positions_list: list[torch.Tensor] = []
    all_trajs: list[list[torch.Tensor]] = []
    all_vals: list[list[float]] = []

    for tr in trackers:
        ts_pos = _extract_ts_guess(
            tr.values, tr.trajectory, tr.settings.filter_passes,
            tr.settings.extraction_criterion, tr.first_coord_reached,
        )
        ts_positions_list.append(ts_pos)
        all_trajs.append(tr.trajectory)
        all_vals.append(tr.values)

    ts_all_positions = torch.cat(ts_positions_list, dim=0)
    ts_state = SimState(
        positions=ts_all_positions,
        masses=state.masses.clone(),
        cell=state.cell.clone(),
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers.clone(),
        system_idx=system_idx.clone(),
    )

    return ts_state, all_trajs, all_vals
