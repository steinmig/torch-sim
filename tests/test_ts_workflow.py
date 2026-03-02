"""Tests for IRC optimizer, geometry optimization, and TS workflow."""

import torch

from torch_sim.hessian import compute_hessian
from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers.irc import (
    IRCSettings,
    batch_irc_optimize,
    gradient_based_converged,
)
from torch_sim.state import SimState, concatenate_states
from torch_sim.workflows.ts_workflow import (
    GeomOptSettings,
    batch_geometry_optimize,
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
