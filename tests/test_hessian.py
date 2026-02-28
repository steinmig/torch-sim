"""Tests for semi-numeric Hessian computation."""

import torch
import pytest

import torch_sim as ts
from torch_sim.state import SimState
from torch_sim.models.interface import ModelInterface
from torch_sim.hessian import compute_hessian


class HarmonicModel(ModelInterface):
    """Simple harmonic potential: E = 0.5 * sum(k * (x - x0)^2).

    Analytical Hessian is k * I.
    """

    def __init__(self, k: float = 1.0, x0: torch.Tensor | None = None,
                 device=None, dtype=torch.float64):
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_stress = False
        self._compute_forces = True
        self.k = k
        self.x0 = x0

    def forward(self, state, **kwargs):
        if not isinstance(state, SimState):
            state = SimState(**state)
        positions = state.positions
        n_systems = state.n_systems
        system_idx = state.system_idx

        if self.x0 is not None:
            x0 = self.x0.to(positions.device, positions.dtype)
        else:
            x0 = torch.zeros_like(positions)

        diff = positions - x0
        per_atom_e = 0.5 * self.k * (diff ** 2).sum(dim=-1)
        energy = torch.zeros(n_systems, device=positions.device, dtype=positions.dtype)
        energy.scatter_add_(0, system_idx, per_atom_e)
        forces = -self.k * diff

        return {"energy": energy, "forces": forces}


def _make_state(n_atoms=3, device=None, dtype=torch.float64):
    device = device or torch.device("cpu")
    positions = torch.randn(n_atoms, 3, device=device, dtype=dtype)
    masses = torch.ones(n_atoms, device=device, dtype=dtype)
    cell = 10.0 * torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    atomic_numbers = torch.ones(n_atoms, device=device, dtype=torch.int64)
    return SimState(
        positions=positions, masses=masses, cell=cell,
        pbc=False, atomic_numbers=atomic_numbers,
    )


class TestComputeHessian:
    @pytest.mark.parametrize("batch", [True, False])
    def test_harmonic_hessian_matches_analytical(self, batch):
        k = 2.5
        n_atoms = 3
        n_dof = 3 * n_atoms
        device = torch.device("cpu")
        dtype = torch.float64

        model = HarmonicModel(k=k, device=device, dtype=dtype)
        state = _make_state(n_atoms=n_atoms, device=device, dtype=dtype)

        hessian = compute_hessian(model, state, delta=0.001, batch_perturbations=batch)

        expected = k * torch.eye(n_dof, device=device, dtype=dtype)
        assert torch.allclose(hessian, expected, atol=1e-6), \
            f"max error: {(hessian - expected).abs().max()}"

    def test_symmetry(self):
        model = HarmonicModel(k=1.0)
        state = _make_state(n_atoms=4)
        hessian = compute_hessian(model, state, delta=0.01)
        assert torch.allclose(hessian, hessian.T, atol=1e-12)

    def test_decreasing_delta_improves_accuracy(self):
        k = 3.0
        n_atoms = 2
        n_dof = 6
        device = torch.device("cpu")
        dtype = torch.float64

        model = HarmonicModel(k=k, device=device, dtype=dtype)
        state = _make_state(n_atoms=n_atoms, device=device, dtype=dtype)

        expected = k * torch.eye(n_dof, device=device, dtype=dtype)

        err_large = (compute_hessian(model, state, delta=0.1) - expected).abs().max()
        err_small = (compute_hessian(model, state, delta=0.001) - expected).abs().max()

        assert err_small < err_large or err_small < 1e-8

    def test_rejects_multi_system(self):
        model = HarmonicModel()
        positions = torch.randn(4, 3)
        masses = torch.ones(4)
        cell = 10.0 * torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        atomic_numbers = torch.ones(4, dtype=torch.int64)
        system_idx = torch.tensor([0, 0, 1, 1])
        state = SimState(
            positions=positions, masses=masses, cell=cell,
            pbc=False, atomic_numbers=atomic_numbers, system_idx=system_idx,
        )
        with pytest.raises(ValueError, match="single-system"):
            compute_hessian(model, state)
