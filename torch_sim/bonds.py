"""Distance-based bond detection for torch-sim.

Detects bonds based on interatomic distances and covalent radii,
following the same logic as SCINE's BondDetector: a bond exists
if d(i,j) < r_cov(i) + r_cov(j) + 0.4 Angstrom.
"""

from __future__ import annotations

import torch

_COVALENT_RADII_ANGSTROM: dict[int, float] | None = None


def _get_covalent_radii() -> dict[int, float]:
    """Lazily load covalent radii from ASE (in Angstrom)."""
    global _COVALENT_RADII_ANGSTROM
    if _COVALENT_RADII_ANGSTROM is None:
        from ase.data import covalent_radii
        _COVALENT_RADII_ANGSTROM = {z: float(covalent_radii[z]) for z in range(len(covalent_radii))}
    return _COVALENT_RADII_ANGSTROM


_TOLERANCE_ANGSTROM = 0.4


def detect_bonds(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    tolerance: float = _TOLERANCE_ANGSTROM,
) -> torch.Tensor:
    """Detect bonds based on interatomic distances and covalent radii.

    A bond between atoms i and j is detected when:
        dist(i, j) < covalent_radius(i) + covalent_radius(j) + tolerance

    Args:
        atomic_numbers: Integer tensor of shape [n_atoms].
        positions: Float tensor of shape [n_atoms, 3] in Angstrom.
        tolerance: Extra distance tolerance in Angstrom (default 0.4).

    Returns:
        Bond order matrix of shape [n_atoms, n_atoms] with 1.0 where a
        bond is detected and 0.0 otherwise. The matrix is symmetric with
        zeros on the diagonal.
    """
    radii_table = _get_covalent_radii()
    n = atomic_numbers.shape[0]
    device = positions.device
    dtype = positions.dtype

    radii = torch.tensor(
        [radii_table.get(int(z), 1.5) for z in atomic_numbers.tolist()],
        device=device, dtype=dtype,
    )

    dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)

    thresholds = radii.unsqueeze(1) + radii.unsqueeze(0) + tolerance

    bonds = (dists < thresholds).to(dtype)
    bonds.fill_diagonal_(0.0)

    return bonds


def get_bond_order(
    bond_matrix: torch.Tensor, i: int, j: int
) -> float:
    """Get the bond order between atoms i and j."""
    return float(bond_matrix[i, j].item())
