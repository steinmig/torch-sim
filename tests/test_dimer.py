"""Tests for the Dimer saddle point optimizer.

Uses the same test surface as the Bofill tests:
    f(x, y) = -cos(x) - 0.5*cos(y)

Saddle points at (pi, 0), (0, pi), etc.
Minima at (0, 0), maxima at (pi, pi).
"""

import math

import pytest
import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers.dimer import (
    DimerSettings,
    batch_dimer_ts_optimize,
    dimer_optimize,
    dimer_ts_optimize,
)
from torch_sim.state import SimState, concatenate_states


def _cosine_eval(params: torch.Tensor) -> tuple[float, torch.Tensor]:
    """f(x,y) = -cos(x) - 0.5*cos(y); returns (value, gradients)."""
    x, y = params[0], params[1]
    value = (-torch.cos(x) - 0.5 * torch.cos(y)).item()
    gradients = torch.stack([torch.sin(x), 0.5 * torch.sin(y)])
    return value, gradients


def _is_saddle(result: torch.Tensor, expected_x: float, expected_y: float, tol: float = 0.05) -> bool:
    return abs(result[0].item() - expected_x) < tol and abs(result[1].item() - expected_y) < tol


_DIMER_SETTINGS = DimerSettings(
    trust_radius=0.2,
    max_iter=500,
    radius=0.01,
    step_max_coeff=1e-5,
    step_rms=1e-6,
    grad_max_coeff=1e-5,
    grad_rms=1e-6,
    delta_value=1e-10,
    convergence_requirement=3,
)


class TestDimer:
    """Dimer optimizer on the cosine surface."""

    def test_saddle_from_0_55pi(self):
        """Near (0.55*pi, 0.55*pi) -> should find a saddle point."""
        params = torch.tensor([0.55 * math.pi, 0.55 * math.pi], dtype=torch.float64)
        result, n_cycles = dimer_optimize(params, _cosine_eval, _DIMER_SETTINGS)
        assert n_cycles < 500
        assert _is_saddle(result, math.pi, 0.0) or _is_saddle(result, 0.0, math.pi), (
            f"Expected saddle at (pi,0) or (0,pi), got ({result[0].item():.4f}, {result[1].item():.4f})"
        )

    def test_saddle_from_0_90pi(self):
        """Near (0.90*pi, 0.10*pi) -> should find saddle at (pi, 0)."""
        params = torch.tensor([0.90 * math.pi, 0.10 * math.pi], dtype=torch.float64)
        result, n_cycles = dimer_optimize(params, _cosine_eval, _DIMER_SETTINGS)
        assert n_cycles < 500
        assert _is_saddle(result, math.pi, 0.0), (
            f"Expected saddle at (pi,0), got ({result[0].item():.4f}, {result[1].item():.4f})"
        )

    def test_with_guess_vector(self):
        """Providing a guess along x should help find the saddle at (pi, 0)."""
        params = torch.tensor([0.90 * math.pi, 0.10 * math.pi], dtype=torch.float64)
        guess = torch.tensor([1.0, 0.0], dtype=torch.float64)
        result, n_cycles = dimer_optimize(params, _cosine_eval, _DIMER_SETTINGS, guess_vector=guess)
        assert n_cycles < 500
        assert _is_saddle(result, math.pi, 0.0), (
            f"Expected saddle at (pi,0), got ({result[0].item():.4f}, {result[1].item():.4f})"
        )

    def test_with_guess_along_y(self):
        """Providing a guess along y should help find the saddle at (0, pi)."""
        params = torch.tensor([0.10 * math.pi, 0.90 * math.pi], dtype=torch.float64)
        guess = torch.tensor([0.0, 1.0], dtype=torch.float64)
        result, n_cycles = dimer_optimize(params, _cosine_eval, _DIMER_SETTINGS, guess_vector=guess)
        assert n_cycles < 500
        assert _is_saddle(result, 0.0, math.pi), (
            f"Expected saddle at (0,pi), got ({result[0].item():.4f}, {result[1].item():.4f})"
        )

    def test_empty_params_raises(self):
        params = torch.tensor([], dtype=torch.float64)
        with pytest.raises(ValueError, match="Empty"):
            dimer_optimize(params, _cosine_eval)

    def test_convergence_reported(self):
        """Should converge in fewer than max_iter cycles."""
        params = torch.tensor([0.90 * math.pi, 0.10 * math.pi], dtype=torch.float64)
        settings = DimerSettings(max_iter=500, radius=0.01)
        result, n_cycles = dimer_optimize(params, _cosine_eval, settings)
        assert n_cycles < 500


# ---------------------------------------------------------------------------
# CosineModel for SimState tests
# ---------------------------------------------------------------------------


class CosineModel(ModelInterface):
    """ModelInterface wrapping f = -cos(x1) - 0.5*cos(x2).

    Treats a 2-"atom" system: atom 0 at (x1, 0, 0) and atom 1 at (x2, 0, 0).
    """

    def __init__(self):
        super().__init__()
        self._compute_stress = False
        self._compute_forces = True

    def forward(self, state, **kwargs):
        if not isinstance(state, SimState):
            state = SimState(**state)

        positions = state.positions
        n_systems = state.n_systems
        system_idx = state.system_idx

        energy = torch.zeros(n_systems, dtype=torch.float64)
        forces = torch.zeros_like(positions)

        for s in range(n_systems):
            mask = system_idx == s
            pos = positions[mask]
            x1, x2 = pos[0, 0], pos[1, 0]
            energy[s] = -torch.cos(x1) - 0.5 * torch.cos(x2)
            idx = mask.nonzero(as_tuple=True)[0]
            forces[idx[0], 0] = -torch.sin(x1)
            forces[idx[1], 0] = -0.5 * torch.sin(x2)

        return {"energy": energy, "forces": forces}


def _make_cosine_state(x1: float, x2: float) -> SimState:
    positions = torch.tensor(
        [[x1, 0.0, 0.0], [x2, 0.0, 0.0]], dtype=torch.float64,
    )
    masses = torch.ones(2, dtype=torch.float64)
    cell = 20.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0)
    atomic_numbers = torch.tensor([1, 1], dtype=torch.int64)
    return SimState(
        positions=positions, masses=masses, cell=cell,
        pbc=False, atomic_numbers=atomic_numbers,
    )


class TestDimerSimState:
    """Tests for the dimer_ts_optimize / batch interface."""

    def test_ts_optimize_finds_saddle(self):
        model = CosineModel()
        state = _make_cosine_state(0.90 * math.pi, 0.10 * math.pi)
        settings = DimerSettings(max_iter=500, radius=0.01)

        ts_state, n_cycles = dimer_ts_optimize(model, state, settings)

        assert n_cycles < 500
        x1 = ts_state.positions[0, 0].item()
        x2 = ts_state.positions[1, 0].item()
        assert abs(x1 - math.pi) < 0.05, f"x1 = {x1}"
        assert abs(x2) < 0.05, f"x2 = {x2}"

    def test_multi_system_raises(self):
        model = CosineModel()
        s1 = _make_cosine_state(1.0, 1.0)
        positions = s1.positions.repeat(2, 1)
        masses = s1.masses.repeat(2)
        cell = s1.cell.repeat(2, 1, 1)
        z = s1.atomic_numbers.repeat(2)
        sidx = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        batched = SimState(
            positions=positions, masses=masses, cell=cell,
            pbc=False, atomic_numbers=z, system_idx=sidx,
        )
        with pytest.raises(ValueError, match="single-system"):
            dimer_ts_optimize(model, batched)

    def test_batch_matches_single(self):
        model = CosineModel()
        x1, x2 = 0.90 * math.pi, 0.10 * math.pi
        settings = DimerSettings(max_iter=500, radius=0.01)

        state_single = _make_cosine_state(x1, x2)
        ts_single, nc_single = dimer_ts_optimize(model, state_single, settings)

        s1 = _make_cosine_state(x1, x2)
        s2 = _make_cosine_state(x1, x2)
        batched = concatenate_states([s1, s2])
        ts_batch, ncs_batch = batch_dimer_ts_optimize(
            model, batched, [settings, settings],
        )

        assert ts_batch.n_systems == 2
        assert nc_single == ncs_batch[0]
        assert nc_single == ncs_batch[1]
        for s in range(2):
            mask = ts_batch.system_idx == s
            assert torch.allclose(
                ts_single.positions, ts_batch.positions[mask], atol=1e-10,
            )

    def test_batch_different_starts(self):
        """Two systems with different starting points should both converge."""
        model = CosineModel()
        s1 = _make_cosine_state(0.90 * math.pi, 0.10 * math.pi)
        s2 = _make_cosine_state(0.10 * math.pi, 0.90 * math.pi)
        batched = concatenate_states([s1, s2])
        settings = DimerSettings(max_iter=500, radius=0.01)

        ts_batch, ncs = batch_dimer_ts_optimize(model, batched, [settings, settings])

        assert ts_batch.n_systems == 2
        for s in range(2):
            assert ncs[s] < 500
            mask = ts_batch.system_idx == s
            pos = ts_batch.positions[mask]
            x_val = pos[0, 0].item()
            y_val = pos[1, 0].item()
            assert (
                _is_saddle(torch.tensor([x_val, y_val]), math.pi, 0.0)
                or _is_saddle(torch.tensor([x_val, y_val]), 0.0, math.pi)
            ), f"System {s}: ({x_val:.4f}, {y_val:.4f})"
