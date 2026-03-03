"""Tests for IRC optimizer, geometry optimization, TS workflow, and memory-safe batching."""

import torch

from torch_sim.hessian import compute_hessian
from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers.bofill import BofillSettings
from torch_sim.optimizers.irc import (
    IRCSettings,
    batch_irc_optimize,
    gradient_based_converged,
)
from torch_sim.optimizers.nt2 import NT2Settings
from torch_sim.state import SimState, concatenate_states
from torch_sim.workflows.ts_workflow import (
    GeomOptSettings,
    TSWorkflowSettings,
    _chunk_indices,
    _extract_range,
    batch_geometry_optimize,
    batch_ts_workflow,
    memory_safe_ts_workflow,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Test model: 1D double-well  E = (x^2 - 1)^2 + y^2 + z^2
# TS at (0,0,0), minima at (±1,0,0)
# ---------------------------------------------------------------------------


class DoubleWellModel(ModelInterface):
    """Per-atom double-well: E = sum_i [(x_i^2 - 1)^2 + y_i^2 + z_i^2]."""

    def __init__(
        self,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ):
        super().__init__()
        self._device = device
        self._dtype = dtype
        self.forward_count = 0

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def forward(self, state: SimState) -> dict[str, torch.Tensor]:
        self.forward_count += 1
        pos = state.positions
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

        e_per_atom = (x**2 - 1) ** 2 + y**2 + z**2
        energy = torch.zeros(
            state.n_systems, device=self.device, dtype=self.dtype
        )
        energy.scatter_add_(0, state.system_idx, e_per_atom)

        fx = -4 * x * (x**2 - 1)
        fy = -2 * y
        fz = -2 * z
        forces = torch.stack([fx, fy, fz], dim=-1)

        return {"energy": energy, "forces": forces}


def _make_dw_state(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    mass: float = 1.0,
) -> SimState:
    return SimState(
        positions=torch.tensor([[x, y, z]], dtype=DTYPE, device=DEVICE),
        masses=torch.tensor([mass], dtype=DTYPE, device=DEVICE),
        cell=100.0 * torch.eye(3, dtype=DTYPE, device=DEVICE).unsqueeze(0),
        pbc=False,
        atomic_numbers=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )


# ========================================================================
# gradient_based_converged
# ========================================================================


class TestGradientBasedConverged:
    def test_all_criteria_met(self):
        g = torch.tensor([1e-6, 1e-6, 1e-6], dtype=DTYPE)
        s = torch.tensor([1e-5, 1e-5, 1e-5], dtype=DTYPE)
        assert gradient_based_converged(
            g, s, 1e-8, 1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 3
        )

    def test_delta_e_too_large(self):
        g = torch.tensor([1e-6, 1e-6, 1e-6], dtype=DTYPE)
        s = torch.tensor([1e-5, 1e-5, 1e-5], dtype=DTYPE)
        assert not gradient_based_converged(
            g, s, 1.0, 1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 3
        )

    def test_too_few_criteria(self):
        g = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
        s = torch.tensor([1e-5, 1e-5, 1e-5], dtype=DTYPE)
        assert not gradient_based_converged(
            g, s, 1e-8, 1e-4, 1e-4, 1e-3, 1e-3, 1e-6, 3
        )


# ========================================================================
# IRC
# ========================================================================


class TestIRC:
    """Tests for batch_irc_optimize."""

    def test_single_system_finds_minima(self):
        model = DoubleWellModel()
        ts = _make_dw_state(0.0, 0.0, 0.0)
        mode = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)

        settings = IRCSettings(
            sd_factor=0.1,
            initial_step_size=0.3,
            max_iter=500,
            grad_max_coeff=1e-3,
            grad_rms=1e-3,
            step_max_coeff=1e-3,
            step_rms=1e-3,
            delta_value=1e-5,
            convergence_requirement=3,
        )

        fwd, bwd, fc, bc = batch_irc_optimize(model, ts, [mode], settings)

        fwd_x = fwd.positions[0, 0].item()
        bwd_x = bwd.positions[0, 0].item()

        assert abs(abs(fwd_x) - 1.0) < 0.1, f"fwd x={fwd_x}"
        assert abs(abs(bwd_x) - 1.0) < 0.1, f"bwd x={bwd_x}"
        assert fwd_x * bwd_x < 0, "forward/backward should be on opposite sides"

    def test_batched_matches_single(self):
        model_s1 = DoubleWellModel()
        model_s2 = DoubleWellModel()

        s1 = _make_dw_state(0.0, 0.0, 0.0, mass=1.0)
        s2 = _make_dw_state(0.0, 0.0, 0.0, mass=4.0)
        mode = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)

        settings = IRCSettings(sd_factor=0.1, initial_step_size=0.3, max_iter=50)

        fwd1, bwd1, _, _ = batch_irc_optimize(model_s1, s1, [mode], settings)
        fwd2, bwd2, _, _ = batch_irc_optimize(model_s2, s2, [mode], settings)
        single_calls_total = model_s1.forward_count + model_s2.forward_count

        model_batch = DoubleWellModel()
        s1 = _make_dw_state(0.0, 0.0, 0.0, mass=1.0)
        s2 = _make_dw_state(0.0, 0.0, 0.0, mass=4.0)
        batched = concatenate_states([s1, s2])
        fwd_b, bwd_b, _, _ = batch_irc_optimize(
            model_batch, batched, [mode, mode], settings
        )
        batch_calls = model_batch.forward_count

        torch.testing.assert_close(fwd_b.positions[0], fwd1.positions[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(fwd_b.positions[1], fwd2.positions[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(bwd_b.positions[0], bwd1.positions[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(bwd_b.positions[1], bwd2.positions[0], atol=1e-4, rtol=1e-4)

        assert batch_calls < single_calls_total, (
            f"batch ({batch_calls}) should use fewer forward calls than "
            f"sum of singles ({single_calls_total})"
        )

    def test_mass_weighting_affects_path(self):
        model = DoubleWellModel()

        light = _make_dw_state(0.0, 0.0, 0.0, mass=1.0)
        heavy = _make_dw_state(0.0, 0.0, 0.0, mass=100.0)
        mode = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)

        settings = IRCSettings(sd_factor=0.1, initial_step_size=0.3, max_iter=10)

        fwd_l, _, _, _ = batch_irc_optimize(model, light, [mode], settings)
        fwd_h, _, _, _ = batch_irc_optimize(model, heavy, [mode], settings)

        disp_l = abs(fwd_l.positions[0, 0].item())
        disp_h = abs(fwd_h.positions[0, 0].item())
        assert disp_l > disp_h, "lighter atom should move further"


# ========================================================================
# IRC starting from Hessian eigenmode
# ========================================================================


class TestIRCWithHessian:
    def test_hessian_mode_irc_pipeline(self):
        model = DoubleWellModel()
        ts = _make_dw_state(0.0, 0.0, 0.0)

        H = compute_hessian(model, ts, delta=0.01)
        evals, evecs = torch.linalg.eigh(H)

        assert evals[0] < 0, f"expected negative eigenvalue, got {evals[0]}"

        mode = evecs[:, 0]
        settings = IRCSettings(
            sd_factor=0.1,
            initial_step_size=0.3,
            max_iter=500,
            grad_max_coeff=1e-3,
            grad_rms=1e-3,
            step_max_coeff=1e-3,
            step_rms=1e-3,
            delta_value=1e-5,
            convergence_requirement=3,
        )

        fwd, bwd, _, _ = batch_irc_optimize(model, ts, [mode], settings)

        fwd_x = fwd.positions[0, 0].item()
        bwd_x = bwd.positions[0, 0].item()

        assert abs(abs(fwd_x) - 1.0) < 0.1
        assert abs(abs(bwd_x) - 1.0) < 0.1
        assert fwd_x * bwd_x < 0


# ========================================================================
# Geometry optimization (IRCOPT)
# ========================================================================


class TestGeomOpt:
    def test_finds_minimum(self):
        model = DoubleWellModel()
        state = _make_dw_state(0.8, 0.1, 0.1)

        settings = GeomOptSettings(
            max_iter=200,
            bfgs_trust_radius=0.2,
            grad_max_coeff=1e-3,
            grad_rms=1e-3,
            step_max_coeff=1e-3,
            step_rms=1e-3,
            delta_value=1e-5,
            convergence_requirement=3,
        )

        result, cycles = batch_geometry_optimize(model, state, settings)

        x = result.positions[0, 0].item()
        y = result.positions[0, 1].item()
        z = result.positions[0, 2].item()

        assert abs(x - 1.0) < 0.05, f"x={x}"
        assert abs(y) < 0.05, f"y={y}"
        assert abs(z) < 0.05, f"z={z}"

    def test_batched_finds_both_minima(self):
        model = DoubleWellModel()

        s1 = _make_dw_state(0.8, 0.1, 0.05)
        s2 = _make_dw_state(-0.8, -0.1, 0.05)
        batched = concatenate_states([s1, s2])

        settings = GeomOptSettings(max_iter=200, bfgs_trust_radius=0.2)

        result, cycles = batch_geometry_optimize(model, batched, settings)

        mask0 = result.system_idx == 0
        x0 = result.positions[mask0][0, 0].item()
        assert abs(x0 - 1.0) < 0.1, f"system 0 x={x0}"

        mask1 = result.system_idx == 1
        x1 = result.positions[mask1][0, 0].item()
        assert abs(x1 + 1.0) < 0.1, f"system 1 x={x1}"


# ========================================================================
# Full Hessian -> IRC -> IRCOPT pipeline
# ========================================================================


class TestPipeline:
    """Hessian -> IRC -> IRCOPT on the double-well."""

    def test_pipeline_finds_minima(self):
        model = DoubleWellModel()
        ts = _make_dw_state(0.0, 0.0, 0.0)

        H = compute_hessian(model, ts, delta=0.01)
        evals, evecs = torch.linalg.eigh(H)
        mode = evecs[:, 0]

        irc_settings = IRCSettings(
            sd_factor=0.1,
            initial_step_size=0.3,
            max_iter=500,
            grad_max_coeff=1e-3,
            grad_rms=1e-3,
            step_max_coeff=1e-3,
            step_rms=1e-3,
            delta_value=1e-5,
            convergence_requirement=3,
        )
        fwd, bwd, _, _ = batch_irc_optimize(model, ts, [mode], irc_settings)

        opt_settings = GeomOptSettings(
            max_iter=200,
            bfgs_trust_radius=0.2,
            grad_max_coeff=1e-3,
            grad_rms=1e-3,
            step_max_coeff=1e-3,
            step_rms=1e-3,
            delta_value=1e-5,
            convergence_requirement=3,
        )

        combined = concatenate_states([fwd, bwd])
        opt_result, _ = batch_geometry_optimize(model, combined, opt_settings)

        mask0 = opt_result.system_idx == 0
        mask1 = opt_result.system_idx == 1
        x_fwd = opt_result.positions[mask0][0, 0].item()
        x_bwd = opt_result.positions[mask1][0, 0].item()

        assert abs(abs(x_fwd) - 1.0) < 0.05, f"fwd opt x={x_fwd}"
        assert abs(abs(x_bwd) - 1.0) < 0.05, f"bwd opt x={x_bwd}"
        assert x_fwd * x_bwd < 0, "should reach opposite minima"

    def test_batched_pipeline(self):
        """Two identical systems through the pipeline should give same result."""
        model = DoubleWellModel()

        s1 = _make_dw_state(0.0, 0.0, 0.0, mass=1.0)
        s2 = _make_dw_state(0.0, 0.0, 0.0, mass=1.0)
        batched = concatenate_states([s1, s2])

        mode = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
        irc_settings = IRCSettings(
            sd_factor=0.1, initial_step_size=0.3, max_iter=50
        )

        fwd, bwd, _, _ = batch_irc_optimize(
            model, batched, [mode, mode], irc_settings
        )

        torch.testing.assert_close(
            fwd.positions[0], fwd.positions[1], msg="fwd systems should match"
        )
        torch.testing.assert_close(
            bwd.positions[0], bwd.positions[1], msg="bwd systems should match"
        )

        opt_settings = GeomOptSettings(max_iter=100, bfgs_trust_radius=0.2)
        combined = concatenate_states([fwd, bwd])
        opt, _ = batch_geometry_optimize(model, combined, opt_settings)

        mask0 = opt.system_idx == 0
        mask1 = opt.system_idx == 1
        torch.testing.assert_close(
            opt.positions[mask0],
            opt.positions[mask1],
            msg="identical systems should optimize identically",
        )


# ========================================================================
# Chunking utilities
# ========================================================================


class TestChunkIndices:
    def test_exact_division(self):
        assert _chunk_indices(6, 3) == [(0, 3), (3, 6)]

    def test_remainder(self):
        assert _chunk_indices(7, 3) == [(0, 3), (3, 6), (6, 7)]

    def test_chunk_larger_than_total(self):
        assert _chunk_indices(3, 10) == [(0, 3)]

    def test_single_element(self):
        assert _chunk_indices(1, 1) == [(0, 1)]

    def test_chunk_size_one(self):
        assert _chunk_indices(3, 1) == [(0, 1), (1, 2), (2, 3)]


class TestExtractRange:
    def test_extract_single(self):
        s1 = _make_dw_state(1.0, 0.0, 0.0)
        s2 = _make_dw_state(2.0, 0.0, 0.0)
        s3 = _make_dw_state(3.0, 0.0, 0.0)
        batched = concatenate_states([s1, s2, s3])

        sub = _extract_range(batched, 1, 2)
        assert sub.n_systems == 1
        assert abs(sub.positions[0, 0].item() - 2.0) < 1e-10

    def test_extract_range(self):
        states = [_make_dw_state(float(i), 0.0, 0.0) for i in range(5)]
        batched = concatenate_states(states)

        sub = _extract_range(batched, 1, 4)
        assert sub.n_systems == 3
        assert abs(sub.positions[0, 0].item() - 1.0) < 1e-10
        assert abs(sub.positions[1, 0].item() - 2.0) < 1e-10
        assert abs(sub.positions[2, 0].item() - 3.0) < 1e-10

    def test_extract_all(self):
        states = [_make_dw_state(float(i), 0.0, 0.0) for i in range(3)]
        batched = concatenate_states(states)

        sub = _extract_range(batched, 0, 3)
        assert sub.n_systems == 3
        torch.testing.assert_close(sub.positions, batched.positions)


# ========================================================================
# Memory-safe TS workflow
# ========================================================================


class TwoAtomDoubleWell(ModelInterface):
    """2-atom double-well for full workflow testing.

    E = sum_i [(x_i^2 - 1)^2 + y_i^2 + z_i^2]
    TS for atom at x=0, minima at x=+-1.
    """

    def __init__(self):
        super().__init__()
        self._device = DEVICE
        self._dtype = DTYPE
        self.forward_count = 0

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def forward(self, state, **kwargs):
        self.forward_count += 1
        if not isinstance(state, SimState):
            state = SimState(**state)
        pos = state.positions
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

        e_per_atom = (x**2 - 1) ** 2 + y**2 + z**2
        energy = torch.zeros(state.n_systems, device=self.device, dtype=self.dtype)
        energy.scatter_add_(0, state.system_idx, e_per_atom)

        fx = -4 * x * (x**2 - 1)
        fy = -2 * y
        fz = -2 * z
        forces = torch.stack([fx, fy, fz], dim=-1)
        return {"energy": energy, "forces": forces}


def _make_2atom_state(x0: float, x1: float) -> SimState:
    """Two hydrogen atoms along x-axis."""
    return SimState(
        positions=torch.tensor([[x0, 0.0, 0.0], [x1, 0.0, 0.0]], dtype=DTYPE, device=DEVICE),
        masses=torch.tensor([1.0, 1.0], dtype=DTYPE, device=DEVICE),
        cell=100.0 * torch.eye(3, dtype=DTYPE, device=DEVICE).unsqueeze(0),
        pbc=False,
        atomic_numbers=torch.tensor([1, 1], dtype=torch.int32, device=DEVICE),
    )


class TestMemorySafeWorkflow:
    """Tests for memory_safe_ts_workflow and its integration."""

    def _workflow_settings(self) -> TSWorkflowSettings:
        return TSWorkflowSettings(
            nt2=NT2Settings(max_iter=3, use_micro_cycles=False),
            tsopt=BofillSettings(
                trust_radius=0.1,
                max_iter=5,
                hessian_update=2,
            ),
            hessian_delta=0.01,
            tsopt_hessian_delta=0.01,
            irc=IRCSettings(
                sd_factor=0.05,
                initial_step_size=0.2,
                max_iter=10,
                grad_max_coeff=1e-2,
                grad_rms=1e-2,
                step_max_coeff=1e-2,
                step_rms=1e-2,
                delta_value=1e-3,
                convergence_requirement=3,
            ),
            ircopt=GeomOptSettings(
                max_iter=10,
                bfgs_trust_radius=0.2,
                grad_max_coeff=1e-2,
                grad_rms=1e-2,
                step_max_coeff=1e-2,
                step_rms=1e-2,
                delta_value=1e-3,
                convergence_requirement=3,
            ),
        )

    def test_single_system_smoke(self):
        """One system through the memory-safe workflow should produce a result."""
        model = TwoAtomDoubleWell()
        state = _make_2atom_state(0.01, 0.9)
        settings = self._workflow_settings()

        result = memory_safe_ts_workflow(
            model, state,
            association_lists=[[]],
            dissociation_lists=[[0, 1]],
            settings=settings,
            max_systems=1,
        )
        assert result.ts_guess.n_systems == 1
        assert result.ts_opt.n_systems == 1
        assert len(result.hessians) == 1
        assert len(result.tsopt_cycles) == 1

    def test_chunked_matches_unchunked(self):
        """Two identical systems: chunked (max_systems=1) should match unchunked."""
        settings = self._workflow_settings()

        model_ref = TwoAtomDoubleWell()
        s1 = _make_2atom_state(0.01, 0.9)
        ref_result = memory_safe_ts_workflow(
            model_ref, s1,
            association_lists=[[]],
            dissociation_lists=[[0, 1]],
            settings=settings,
            max_systems=10,
        )

        model_chunk = TwoAtomDoubleWell()
        s1a = _make_2atom_state(0.01, 0.9)
        s1b = _make_2atom_state(0.01, 0.9)
        batched = concatenate_states([s1a, s1b])
        chunk_result = memory_safe_ts_workflow(
            model_chunk, batched,
            association_lists=[[], []],
            dissociation_lists=[[0, 1], [0, 1]],
            settings=settings,
            max_systems=1,
        )

        assert chunk_result.ts_guess.n_systems == 2
        assert chunk_result.ts_opt.n_systems == 2

        mask0 = chunk_result.ts_opt.system_idx == 0
        mask1 = chunk_result.ts_opt.system_idx == 1
        torch.testing.assert_close(
            chunk_result.ts_opt.positions[mask0],
            ref_result.ts_opt.positions,
            atol=1e-6, rtol=1e-6,
        )
        torch.testing.assert_close(
            chunk_result.ts_opt.positions[mask0],
            chunk_result.ts_opt.positions[mask1],
            atol=1e-6, rtol=1e-6,
        )

    def test_hessian_filter_removes_systems(self):
        """Systems without imaginary frequency should be filtered before IRC."""
        model = TwoAtomDoubleWell()
        s_near_ts = _make_2atom_state(0.01, 0.01)
        s_at_min = _make_2atom_state(1.0, 1.0)
        batched = concatenate_states([s_near_ts, s_at_min])

        settings = self._workflow_settings()
        settings.nt2 = NT2Settings(max_iter=1, use_micro_cycles=False)

        result = memory_safe_ts_workflow(
            model, batched,
            association_lists=[[], []],
            dissociation_lists=[[0, 1], [0, 1]],
            settings=settings,
            max_systems=10,
        )

        assert result.ts_guess.n_systems == 2
        assert result.ts_opt.n_systems == 2
        assert len(result.hessians) == 2

        n_neg = sum(1 for ev in result.eigenvalues if ev[0] < 0)
        assert result.irc_forward.n_systems == n_neg
        assert result.irc_backward.n_systems == n_neg

    def test_four_systems_chunk_two(self):
        """Four systems processed in chunks of 2."""
        model = TwoAtomDoubleWell()
        states = [_make_2atom_state(0.01, 0.9) for _ in range(4)]
        batched = concatenate_states(states)

        settings = self._workflow_settings()

        result = memory_safe_ts_workflow(
            model, batched,
            association_lists=[[]] * 4,
            dissociation_lists=[[0, 1]] * 4,
            settings=settings,
            max_systems=2,
        )

        assert result.ts_guess.n_systems == 4
        assert result.ts_opt.n_systems == 4
        assert len(result.tsopt_cycles) == 4
        assert len(result.nt2_trajectories) == 4
        assert len(result.nt2_energies) == 4
