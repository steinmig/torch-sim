"""Tests for the Bofill saddle point optimizer.

Ported from SCINE Utils OptimizerTest.cpp (BofillTest_1 through BofillTest_4).

Uses the same test surface as SCINE utilities:
    f(x, y) = -cos(x) - 0.5*cos(y)

Saddle points at (pi, 0), (0, pi), etc.
Minima at (0, 0), maxima at (pi, pi).
"""

import math

import pytest
import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers.bofill import (
    BofillSettings,
    batch_bofill_ts_optimize,
    bofill_optimize,
    bofill_ts_optimize,
)
from torch_sim.state import SimState, concatenate_states
from torch_sim import concatenate_states


def _cosine_surface(params: torch.Tensor, hessian_update: bool):
    """f(x,y) = -cos(x) - 0.5*cos(y)."""
    x, y = params[0], params[1]
    value = -torch.cos(x) - 0.5 * torch.cos(y)
    gradients = torch.stack([torch.sin(x), 0.5 * torch.sin(y)])

    if hessian_update:
        hessian = torch.zeros(2, 2, device=params.device, dtype=params.dtype)
        hessian[0, 0] = torch.cos(x)
        hessian[1, 1] = 0.5 * torch.cos(y)
    else:
        hessian = torch.eye(2, device=params.device, dtype=params.dtype)

    return value.item(), gradients, hessian


# SCINE default thresholds (GradientBasedCheck defaults) with
# tighter overrides matching the SCINE BofillTest settings.
_BOFILL_SETTINGS = BofillSettings(
    trust_radius=0.1,
    hessian_update=5,
    max_iter=100,
    step_max_coeff=1e-7,
    step_rms=1e-8,
    grad_max_coeff=1e-7,
    grad_rms=1e-8,
    delta_value=1e-12,
    convergence_requirement=4,
)


class TestBofill:
    """Ported from SCINE OptimizerTest: BofillTest_1 through BofillTest_4."""

    def test_saddle_from_0_55pi(self):
        """BofillTest_1: starting from (0.55*pi, 0.55*pi) → saddle at (pi, 0)."""
        params = torch.tensor(
            [0.55 * math.pi, 0.55 * math.pi], dtype=torch.float64,
        )
        result, n_cycles = bofill_optimize(params, _cosine_surface, _BOFILL_SETTINGS)
        assert n_cycles < 100
        assert abs(result[0].item() - math.pi) < 1e-6, f"x = {result[0].item()}"
        assert abs(result[1].item()) < 1e-6, f"y = {result[1].item()}"

    def test_saddle_from_0_51pi(self):
        """BofillTest_2: starting from (0.51*pi, 0.49*pi) → saddle at (pi, 0)."""
        params = torch.tensor(
            [0.51 * math.pi, 0.49 * math.pi], dtype=torch.float64,
        )
        result, n_cycles = bofill_optimize(params, _cosine_surface, _BOFILL_SETTINGS)
        assert n_cycles < 100
        assert abs(result[0].item() - math.pi) < 1e-6
        assert abs(result[1].item()) < 1e-6

    def test_saddle_from_0_55_0_45(self):
        """BofillTest_3: starting from (0.55*pi, 0.45*pi) → saddle at (pi, 0)."""
        params = torch.tensor(
            [0.55 * math.pi, 0.45 * math.pi], dtype=torch.float64,
        )
        result, n_cycles = bofill_optimize(params, _cosine_surface, _BOFILL_SETTINGS)
        assert n_cycles < 100
        assert abs(result[0].item() - math.pi) < 1e-6
        assert abs(result[1].item()) < 1e-6

    def test_saddle_from_0_90pi(self):
        """BofillTest_4: starting from (0.90*pi, 0.10*pi) → saddle at (pi, 0)."""
        params = torch.tensor(
            [0.90 * math.pi, 0.10 * math.pi], dtype=torch.float64,
        )
        result, n_cycles = bofill_optimize(params, _cosine_surface, _BOFILL_SETTINGS)
        assert n_cycles < 100
        assert abs(result[0].item() - math.pi) < 1e-6
        assert abs(result[1].item()) < 1e-6

    def test_empty_params_raises(self):
        params = torch.tensor([], dtype=torch.float64)
        with pytest.raises(ValueError, match="Empty"):
            bofill_optimize(params, _cosine_surface)


# ---------------------------------------------------------------------------
# CosineModel: wraps the cosine surface as a ModelInterface for SimState tests
# ---------------------------------------------------------------------------

class CosineModel(ModelInterface):
    """ModelInterface wrapping f = -cos(x1) - 0.5*cos(x2).

    Treats a 2-"atom" system: atom 0 at (x1, 0, 0) and atom 1 at (x2, 0, 0).
    Only the x-coordinates matter; y/z are ignored.
    """

    def __init__(self):
        super().__init__()
        self._compute_stress = False
        self._compute_forces = True
        self.forward_count = 0

    def forward(self, state, **kwargs):
        self.forward_count += 1
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
    """Two-atom state where x-coords map to the 2D surface."""
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


class TestBofillSimState:
    """Tests for the bofill_ts_optimize / batch interface."""

    def test_ts_optimize_finds_saddle(self):
        """bofill_ts_optimize should find the saddle using finite-diff Hessian."""
        model = CosineModel()
        state = _make_cosine_state(0.55 * math.pi, 0.55 * math.pi)
        settings = BofillSettings(
            trust_radius=0.1,
            hessian_update=5,
            max_iter=150,
            delta_value=1e-8,
        )

        ts_state, n_cycles = bofill_ts_optimize(
            model, state, settings, hessian_delta=0.001,
        )

        assert n_cycles < 150
        x1 = ts_state.positions[0, 0].item()
        x2 = ts_state.positions[1, 0].item()
        assert abs(x1 - math.pi) < 1e-3, f"x1 = {x1}"
        assert abs(x2) < 1e-3, f"x2 = {x2}"

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
            bofill_ts_optimize(model, batched)

    def test_batch_matches_single(self):
        """Two identical copies batched should each match single-system result."""
        model_single = CosineModel()
        x1, x2 = 0.90 * math.pi, 0.10 * math.pi
        settings = BofillSettings(
            trust_radius=0.1, hessian_update=5, max_iter=100,
        )

        state_single = _make_cosine_state(x1, x2)
        ts_single, nc_single = bofill_ts_optimize(
            model_single, state_single, settings, hessian_delta=0.001,
        )
        single_calls = model_single.forward_count

        model_batch = CosineModel()
        s1 = _make_cosine_state(x1, x2)
        s2 = _make_cosine_state(x1, x2)
        batched = concatenate_states([s1, s2])
        ts_batch, ncs_batch = batch_bofill_ts_optimize(
            model_batch, batched, [settings, settings], hessian_delta=0.001,
        )
        batch_calls = model_batch.forward_count

        assert ts_batch.n_systems == 2
        assert nc_single == ncs_batch[0]
        assert nc_single == ncs_batch[1]
        for s in range(2):
            mask = ts_batch.system_idx == s
            assert torch.allclose(
                ts_single.positions, ts_batch.positions[mask], atol=1e-10,
            )

        n_systems = 2
        assert batch_calls < n_systems * single_calls, (
            f"batch ({batch_calls}) should use fewer forward calls than "
            f"{n_systems} x single ({single_calls})"
        )
