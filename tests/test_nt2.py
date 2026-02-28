"""Tests for the NT2 (Newton Trajectory 2) optimizer.

Includes a simple diatomic mock model, plus a multi-atom LJ+Gaussian model
(port of SCINE's TestCalculator) for realistic SN2 reaction tests.
"""

import math
import torch
import pytest

from ase.data import covalent_radii as _ase_cov_radii

from torch_sim.state import SimState
from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers.nt2 import (
    nt2_optimize,
    batch_nt2_optimize,
    NT2Settings,
    _build_reactive_atoms,
    _build_constraints_map,
    _center_to_center,
    _connected_nuclei,
    _smallest_cov_radius,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOHR_TO_ANGSTROM = 0.5291772105638411
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Atomic numbers: Cl=17, C=6, H=1, Br=35
_SN2_ELEMENTS = [17, 6, 1, 1, 1, 35]

# SCINE SN2 starting positions in Bohr (from NtOptimizer2Test.cpp)
_SN2_POSITIONS_BOHR = [
    [3.7376961460e+00, 2.1020866350e-04, 4.5337439168e-02],   # Cl
    [-3.8767703481e+00, -2.4803422157e-05, -1.2049608882e-01], # C
    [-2.3620148614e+00, 1.3238308540e+00, 1.0376490681e-01],   # H
    [-2.3809041075e+00, -8.2773666259e-01, 9.6331578315e-01],  # H
    [-2.3309449521e+00, -4.9652606314e-01, -1.3293307598e+00], # H
    [-7.4798903722e+00, 2.6536371103e-04, -1.9897114399e-01],  # Br
]

# Converted to Angstrom for non-SCINE tests
_SN2_POSITIONS_ANG = [
    [x * BOHR_TO_ANGSTROM for x in row] for row in _SN2_POSITIONS_BOHR
]

# SCINE ElementInfo covalent radii in Bohr (from ElementData.cpp: pm → Å → Bohr)
# Used for reaction-coordinate force scaling (NtUtils::smallestCovalentRadius)
# and for the LJ+Gaussian potential (TestCalculator).
_SCINE_COV_RADII_BOHR: dict[int, float] = {
    1:  32 / 100 * ANGSTROM_TO_BOHR,   # H:  32 pm
    6:  75 / 100 * ANGSTROM_TO_BOHR,   # C:  75 pm
    17: 100 / 100 * ANGSTROM_TO_BOHR,  # Cl: 100 pm
    35: 117 / 100 * ANGSTROM_TO_BOHR,  # Br: 117 pm
}

# SCINE BondDetectorRadii (CSD – Cambridge Structural Database) in Bohr.
# Used by BondDetector::detectBonds for bond existence checks.
_SCINE_BOND_DET_RADII_BOHR: dict[int, float] = {
    1:  0.23 * ANGSTROM_TO_BOHR,   # H
    6:  0.68 * ANGSTROM_TO_BOHR,   # C
    17: 0.99 * ANGSTROM_TO_BOHR,   # Cl
    35: 1.21 * ANGSTROM_TO_BOHR,   # Br
}

# SCINE bond detection tolerance: toBohr(Angstrom(0.4))
_SCINE_BOND_TOLERANCE_BOHR = 0.4 * ANGSTROM_TO_BOHR


# ---------------------------------------------------------------------------
# DiatomicMockModel  (simple quartic PES for 2-atom tests)
# ---------------------------------------------------------------------------

class DiatomicMockModel(ModelInterface):
    """Two-atom model with PES: E(r) = (r-8)^2 * (r-4) * (r-12).

    Has minima near r=4 and r=12, saddle near r=8.
    Handles batched (multi-system) states automatically.
    """

    def __init__(self, device=None, dtype=torch.float64):
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_stress = False
        self._compute_forces = True

    def forward(self, state, **kwargs):
        if not isinstance(state, SimState):
            state = SimState(**state)

        positions = state.positions
        n_systems = state.n_systems
        system_idx = state.system_idx

        energy = torch.zeros(n_systems, device=self._device, dtype=self._dtype)
        forces = torch.zeros_like(positions)

        for s in range(n_systems):
            mask = system_idx == s
            pos = positions[mask]
            p1, p2 = pos[0], pos[1]
            v12 = p1 - p2
            r12 = v12.norm()

            e = (r12 - 8.0) ** 2 * (r12 - 4.0) * (r12 - 12.0)
            de_dr = 4.0 * (r12 ** 3 - 24.0 * r12 ** 2 + 184.0 * r12 - 448.0)

            grad_vec = de_dr / max(r12.item(), 1e-12) * v12
            energy[s] = e

            idx = mask.nonzero(as_tuple=True)[0]
            forces[idx[0]] = -grad_vec
            forces[idx[1]] = grad_vec

        return {"energy": energy, "forces": forces}


# ---------------------------------------------------------------------------
# LJGaussianModel  (port of SCINE TestCalculator)
# ---------------------------------------------------------------------------

class LJGaussianModel(ModelInterface):
    """All-pairs LJ + Gaussian PES, a faithful port of the SCINE TestCalculator.

    Energy for each pair (i, j):
        E = well_depth * (lj12 - 2*lj6) + E_gauss
    where
        rMin   = cov_radius[i] + cov_radius[j]
        lj     = rMin / r
        scaling = min(rMin / 2, 2.0)
        well_depth = 0.2 * scaling
        u      = (r - 2.5*scaling) / scaling
        E_gauss = (0.4 * scaling / r) * exp(-u^2)

    When *cov_radii* is provided, those radii are used directly; otherwise
    ASE covalent radii (Angstrom) are used.  The gradient formula matches
    the one in SCINE's TestCalculator.cpp exactly so that numerical results
    are reproducible.

    Handles batched (multi-system) states.
    """

    def __init__(self, device=None, dtype=torch.float64,
                 cov_radii: dict[int, float] | None = None,
                 use_scine_gradient: bool = False):
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_stress = False
        self._compute_forces = True
        self._cov_radii = cov_radii
        self._use_scine_gradient = use_scine_gradient

    def _get_radius(self, z: int) -> float:
        if self._cov_radii is not None:
            return self._cov_radii[z]
        return float(_ase_cov_radii[z])

    def forward(self, state, **kwargs):
        if not isinstance(state, SimState):
            state = SimState(**state)

        positions = state.positions
        atomic_numbers = state.atomic_numbers
        n_systems = state.n_systems
        system_idx = state.system_idx

        energy = torch.zeros(n_systems, device=self._device, dtype=self._dtype)
        forces = torch.zeros_like(positions)

        for s in range(n_systems):
            mask = system_idx == s
            pos_s = positions[mask]
            z_s = atomic_numbers[mask]
            n = pos_s.shape[0]
            idx_global = mask.nonzero(as_tuple=True)[0]

            e_sys = torch.tensor(0.0, device=self._device, dtype=self._dtype)

            for i in range(n):
                rad_i = self._get_radius(int(z_s[i].item()))
                for j in range(i):
                    r_vec = pos_s[i] - pos_s[j]
                    dist = r_vec.norm()
                    r = dist.item()
                    if r < 1e-14:
                        continue

                    rad_j = self._get_radius(int(z_s[j].item()))
                    r_min = rad_i + rad_j
                    scaling = min(r_min / 2.0, 2.0)
                    well_depth = 0.2 * scaling

                    lj = r_min / r
                    lj6 = lj ** 6
                    lj12 = lj6 ** 2

                    u = (r - 2.5 * scaling) / scaling
                    e_gauss = (0.4 * scaling / r) * math.exp(-u * u)

                    e_pair = well_depth * (lj12 - 2.0 * lj6) + e_gauss
                    e_sys = e_sys + e_pair

                    # LJ derivative: dE_lj/dr
                    de_lj = well_depth * 12.0 * (lj6 / r - lj12 / r)

                    if self._use_scine_gradient:
                        # SCINE TestCalculator.cpp formula (has approximation
                        # where scaling^2 is replaced by 1, valid when
                        # scaling ≈ 1 in Bohr)
                        gderiv = -(-5.0 * scaling * r + 2.0 * r * r + 1.0) / r
                        de_gauss = gderiv * e_gauss
                    else:
                        # Exact analytical derivative
                        s2 = scaling * scaling
                        de_gauss = e_gauss * (5.0 * scaling * r - 2.0 * r * r - s2) / (r * s2)

                    de_dr = de_lj + de_gauss

                    grad_contrib = (de_dr / r) * r_vec
                    forces[idx_global[i]] -= grad_contrib
                    forces[idx_global[j]] += grad_contrib

            energy[s] = e_sys

        return {"energy": energy, "forces": forces}


# ---------------------------------------------------------------------------
# State construction helpers
# ---------------------------------------------------------------------------

def _make_diatomic_state(r=11.0, device=None, dtype=torch.float64):
    """Two H atoms separated by distance r along y-axis."""
    device = device or torch.device("cpu")
    positions = torch.tensor([[0.0, r, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype)
    masses = torch.ones(2, device=device, dtype=dtype)
    cell = 20.0 * torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    atomic_numbers = torch.tensor([1, 1], device=device, dtype=torch.int64)
    return SimState(positions=positions, masses=masses, cell=cell,
                    pbc=False, atomic_numbers=atomic_numbers)


def _make_batched_diatomic_state(rs: list[float], device=None, dtype=torch.float64):
    """N pairs of H atoms, each separated by the corresponding distance."""
    device = device or torch.device("cpu")
    n = len(rs)
    positions_list = []
    for r in rs:
        positions_list.append(torch.tensor([[0.0, r, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype))
    positions = torch.cat(positions_list, dim=0)
    masses = torch.ones(2 * n, device=device, dtype=dtype)
    cell = 20.0 * torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)
    atomic_numbers = torch.ones(2 * n, device=device, dtype=torch.int64)
    system_idx = torch.arange(n, device=device, dtype=torch.int64).repeat_interleave(2)
    return SimState(
        positions=positions, masses=masses, cell=cell,
        pbc=False, atomic_numbers=atomic_numbers, system_idx=system_idx,
    )


def _make_sn2_state(device=None, dtype=torch.float64, bohr: bool = False):
    """Cl + CH3Br system for SN2 NT2 tests.

    Args:
        bohr: If True, positions are in Bohr (for SCINE-matching tests).
              If False, positions are in Angstrom.
    """
    device = device or torch.device("cpu")
    pos_data = _SN2_POSITIONS_BOHR if bohr else _SN2_POSITIONS_ANG
    positions = torch.tensor(pos_data, device=device, dtype=dtype)
    n_atoms = positions.shape[0]
    masses = torch.ones(n_atoms, device=device, dtype=dtype)
    cell_size = 30.0 if bohr else 30.0 * BOHR_TO_ANGSTROM
    cell = cell_size * torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    atomic_numbers = torch.tensor(_SN2_ELEMENTS, device=device, dtype=torch.int64)
    return SimState(positions=positions, masses=masses, cell=cell,
                    pbc=False, atomic_numbers=atomic_numbers)


def _make_batched_sn2_state(n_copies: int, device=None, dtype=torch.float64,
                            bohr: bool = False):
    """N copies of the SN2 system as a batched SimState."""
    device = device or torch.device("cpu")
    single = _make_sn2_state(device=device, dtype=dtype, bohr=bohr)
    n_atoms = single.positions.shape[0]
    positions = single.positions.repeat(n_copies, 1)
    masses = single.masses.repeat(n_copies)
    cell = single.cell.repeat(n_copies, 1, 1)
    atomic_numbers = single.atomic_numbers.repeat(n_copies)
    system_idx = torch.arange(n_copies, device=device, dtype=torch.int64).repeat_interleave(n_atoms)
    return SimState(
        positions=positions, masses=masses, cell=cell,
        pbc=False, atomic_numbers=atomic_numbers, system_idx=system_idx,
    )


# ---- shared settings ----

_ASSOC_SETTINGS = NT2Settings(
    total_force_norm=0.1,
    sd_factor=1.0,
    max_iter=200,
    attractive_distance_stop=0.5,
    attractive_bond_order_stop=0.75,
    use_micro_cycles=False,
    filter_passes=10,
    extraction_criterion="highest",
)

_SN2_SETTINGS = NT2Settings(
    total_force_norm=0.1,
    sd_factor=1.0,
    max_iter=500,
    attractive_distance_stop=0.9,
    attractive_bond_order_stop=0.75,
    repulsive_bond_order_stop=0.15,
    use_micro_cycles=False,
    filter_passes=10,
    extraction_criterion="lastBeforeTarget",
    extra_macrocycles_after_bond_criteria=10,
)

_SN2_SETTINGS_BOHR = NT2Settings(
    total_force_norm=0.1,
    sd_factor=1.0,
    max_iter=500,
    attractive_distance_stop=0.9,
    attractive_bond_order_stop=0.75,
    repulsive_bond_order_stop=0.15,
    use_micro_cycles=True,
    number_of_micro_cycles=10,
    filter_passes=10,
    extraction_criterion="lastBeforeTarget",
    extra_macrocycles_after_bond_criteria=0,
    cov_radii=_SCINE_COV_RADII_BOHR,
    bond_det_cov_radii=_SCINE_BOND_DET_RADII_BOHR,
    bond_tolerance=_SCINE_BOND_TOLERANCE_BOHR,
)


# ===================================================================
# Tests: helper functions (ported from NtOptimizer2Test.cpp)
# ===================================================================

class TestNT2Helpers:
    """Unit tests for low-level helpers, mirroring SCINE tests."""

    def test_center_to_center(self):
        """Port of NtOptimizer2Tests.Center2CenterWorks."""
        pos = torch.tensor([
            [-1.0,  0.0,  0.0],
            [ 1.0,  2.0, 10.0],
            [ 1.0, -2.0, 10.0],
            [ 4.2,  3.1, -0.9],
        ], dtype=torch.float64)

        lhs = [0]
        rhs = [1, 2]
        c2c = _center_to_center(pos, lhs, rhs)
        expected = torch.tensor([-2.0, 0.0, -10.0], dtype=torch.float64)
        assert torch.allclose(c2c, expected, atol=1e-12)

    def test_smallest_cov_radius(self):
        """Port of NtOptimizer2Tests.SmallestCovRadiusWorks."""
        z = torch.tensor(_SN2_ELEMENTS, dtype=torch.int64)  # Cl, C, H, H, H, Br

        rad_ClCBr = _smallest_cov_radius(z, [0, 1, 5])
        assert abs(rad_ClCBr - _ase_cov_radii[6]) < 1e-10, "Should be C radius"

        rad_CHH = _smallest_cov_radius(z, [1, 2, 3])
        assert abs(rad_CHH - _ase_cov_radii[1]) < 1e-10, "Should be H radius"

    def test_connected_nuclei_single_component(self):
        """All indices bonded → single component."""
        bond_matrix = torch.zeros(5, 5, dtype=torch.float64)
        bond_matrix[0, 2] = bond_matrix[2, 0] = 1.0
        bond_matrix[2, 4] = bond_matrix[4, 2] = 1.0
        components = _connected_nuclei([0, 2, 4], bond_matrix)
        assert len(components) == 1
        assert sorted(components[0]) == [0, 2, 4]

    def test_connected_nuclei_two_components(self):
        """Disjoint indices → separate components."""
        bond_matrix = torch.zeros(10, 10, dtype=torch.float64)
        components = _connected_nuclei([1, 9], bond_matrix)
        assert len(components) == 2
        flat = sorted([c[0] for c in components])
        assert flat == [1, 9]

    def test_reactive_atoms_list(self):
        atoms = _build_reactive_atoms([0, 1], [1, 5])
        assert atoms == [0, 1, 5]

    def test_constraints_map(self):
        """Port of SCINE check: atom 1 participates in assoc(0,1) and dissoc(1,5)."""
        reactive = _build_reactive_atoms([0, 1], [1, 5])
        cmap = _build_constraints_map(6, reactive, [0, 1], [1, 5])
        assert cmap[1] == [0, 1, 1, 5]


# ===================================================================
# Tests: single-system diatomic
# ===================================================================

class TestNT2SingleSystem:
    def test_empty_lists_raises(self):
        model = DiatomicMockModel()
        state = _make_diatomic_state()
        with pytest.raises(ValueError, match="non-empty"):
            nt2_optimize(model, state, [], [])

    def test_multi_system_raises_on_single_api(self):
        model = DiatomicMockModel()
        state = _make_batched_diatomic_state([11.0, 10.0])
        with pytest.raises(ValueError, match="single-system"):
            nt2_optimize(model, state, [0, 1], [])

    def test_association_finds_ts_guess(self):
        model = DiatomicMockModel()
        state = _make_diatomic_state(r=11.0)

        ts_state, traj, vals = nt2_optimize(
            model, state,
            association_list=[0, 1],
            dissociation_list=[],
            settings=_ASSOC_SETTINGS,
        )

        assert len(traj) > 1, "Should have run at least 2 cycles"
        assert len(vals) == len(traj)
        assert ts_state.n_systems == 1

        r_ts = (ts_state.positions[0] - ts_state.positions[1]).norm().item()
        assert 5.0 < r_ts < 11.0, f"TS guess distance {r_ts} should be between minima"

    def test_trajectory_decreasing_distance(self):
        model = DiatomicMockModel()
        state = _make_diatomic_state(r=11.0)

        settings = NT2Settings(
            total_force_norm=0.15,
            sd_factor=1.0,
            max_iter=50,
            attractive_distance_stop=0.3,
            use_micro_cycles=False,
            extraction_criterion="highest",
        )

        _, traj, _ = nt2_optimize(
            model, state,
            association_list=[0, 1],
            dissociation_list=[],
            settings=settings,
        )

        initial_dist = (traj[0][0] - traj[0][1]).norm().item()
        final_dist = (traj[-1][0] - traj[-1][1]).norm().item()
        assert final_dist < initial_dist, "Distance should decrease during association"


# ===================================================================
# Tests: SN2 with LJGaussianModel (port of SCINE SN2 tests)
# ===================================================================

class TestNT2SN2:
    """SN2 reaction tests using the LJGaussianModel (SCINE TestCalculator port).

    System: Cl + CH3-Br
    Association: Cl -> C  (atoms 0, 1)
    Dissociation: C -> Br (atoms 1, 5)
    """

    def test_sn2_association_dissociation(self):
        """Cl-C distance should decrease, C-Br distance should increase."""
        model = LJGaussianModel()
        state = _make_sn2_state()
        pos0 = state.positions.clone()
        initial_cl_c = (pos0[0] - pos0[1]).norm().item()
        initial_c_br = (pos0[1] - pos0[5]).norm().item()

        ts_state, traj, vals = nt2_optimize(
            model, state,
            association_list=[0, 1],
            dissociation_list=[1, 5],
            settings=_SN2_SETTINGS,
        )

        assert len(traj) > 1, "Should have run multiple cycles"

        final_pos = traj[-1]
        final_cl_c = (final_pos[0] - final_pos[1]).norm().item()
        final_c_br = (final_pos[1] - final_pos[5]).norm().item()

        assert final_cl_c < initial_cl_c, (
            f"Cl-C should decrease: {initial_cl_c:.3f} -> {final_cl_c:.3f}"
        )
        assert final_c_br > initial_c_br, (
            f"C-Br should increase: {initial_c_br:.3f} -> {final_c_br:.3f}"
        )

    def test_sn2_ts_guess_reasonable(self):
        """TS guess should have Cl-C shorter than start and C-Br longer than start."""
        model = LJGaussianModel()
        state = _make_sn2_state()
        pos0 = state.positions.clone()

        ts_state, _, vals = nt2_optimize(
            model, state,
            association_list=[0, 1],
            dissociation_list=[1, 5],
            settings=_SN2_SETTINGS,
        )

        ts_cl_c = (ts_state.positions[0] - ts_state.positions[1]).norm().item()
        ts_c_br = (ts_state.positions[1] - ts_state.positions[5]).norm().item()
        init_cl_c = (pos0[0] - pos0[1]).norm().item()
        init_c_br = (pos0[1] - pos0[5]).norm().item()

        assert ts_cl_c < init_cl_c, "TS Cl-C should be shorter than initial"
        assert ts_c_br > init_c_br, "TS C-Br should be longer than initial"

    def test_sn2_ts_guess_bond_lengths(self):
        """Check TS guess bond lengths against SCINE reference values.

        Runs the NT2 optimization in Bohr with SCINE covalent radii and the
        SCINE gradient formula so the PES matches exactly.  The TS guess
        bond lengths (computed in Bohr, converted to Angstrom) are compared
        against the reference from NtOptimizer2Test.cpp:
            Cl-C  (0-1): 4.5981931756 Bohr
            Cl-Br (0-5): 9.8883569663 Bohr
            C-H   (1-4): 2.0063532273 Bohr
            C-Br  (1-5): 5.2901692863 Bohr
        """
        ref_cl_c_ang = 4.5981931756e+00 * BOHR_TO_ANGSTROM
        ref_cl_br_ang = 9.8883569663e+00 * BOHR_TO_ANGSTROM
        ref_c_h_ang = 2.0063532273e+00 * BOHR_TO_ANGSTROM
        ref_c_br_ang = 5.2901692863e+00 * BOHR_TO_ANGSTROM

        model = LJGaussianModel(
            cov_radii=_SCINE_COV_RADII_BOHR, use_scine_gradient=True,
        )
        state = _make_sn2_state(bohr=True)

        ts_state, _, _ = nt2_optimize(
            model, state,
            association_list=[0, 1],
            dissociation_list=[1, 5],
            settings=_SN2_SETTINGS_BOHR,
        )

        pos = ts_state.positions
        ts_cl_c = (pos[0] - pos[1]).norm().item() * BOHR_TO_ANGSTROM
        ts_cl_br = (pos[0] - pos[5]).norm().item() * BOHR_TO_ANGSTROM
        ts_c_h = (pos[1] - pos[4]).norm().item() * BOHR_TO_ANGSTROM
        ts_c_br = (pos[1] - pos[5]).norm().item() * BOHR_TO_ANGSTROM

        tol = 1e-3 * BOHR_TO_ANGSTROM  # SCINE uses EXPECT_NEAR(..., 1e-3) in Bohr
        assert abs(ts_cl_c - ref_cl_c_ang) < tol, (
            f"Cl-C: {ts_cl_c:.6f} vs ref {ref_cl_c_ang:.6f} "
            f"(diff={abs(ts_cl_c - ref_cl_c_ang):.6f})"
        )
        assert abs(ts_cl_br - ref_cl_br_ang) < tol, (
            f"Cl-Br: {ts_cl_br:.6f} vs ref {ref_cl_br_ang:.6f} "
            f"(diff={abs(ts_cl_br - ref_cl_br_ang):.6f})"
        )
        assert abs(ts_c_h - ref_c_h_ang) < tol, (
            f"C-H: {ts_c_h:.6f} vs ref {ref_c_h_ang:.6f} "
            f"(diff={abs(ts_c_h - ref_c_h_ang):.6f})"
        )
        assert abs(ts_c_br - ref_c_br_ang) < tol, (
            f"C-Br: {ts_c_br:.6f} vs ref {ref_c_br_ang:.6f} "
            f"(diff={abs(ts_c_br - ref_c_br_ang):.6f})"
        )

    def test_sn2_extraction_criterion_highest(self):
        """With 'highest' criterion, the TS guess should be the highest-energy point."""
        model = LJGaussianModel()
        state = _make_sn2_state()

        settings = NT2Settings(**{**_SN2_SETTINGS.__dict__, "extraction_criterion": "highest"})
        ts_state, _, vals = nt2_optimize(
            model, state, [0, 1], [1, 5], settings,
        )

        ts_pos = ts_state.positions
        assert ts_pos.shape == (6, 3)
        assert torch.isfinite(ts_pos).all()

    def test_sn2_extraction_criterion_first(self):
        """With 'first' criterion, should still find a valid TS guess."""
        model = LJGaussianModel()
        state = _make_sn2_state()

        settings = NT2Settings(**{**_SN2_SETTINGS.__dict__, "extraction_criterion": "first"})
        ts_state, _, vals = nt2_optimize(
            model, state, [0, 1], [1, 5], settings,
        )

        ts_pos = ts_state.positions
        assert ts_pos.shape == (6, 3)
        assert torch.isfinite(ts_pos).all()

    def test_sn2_extra_macrocycles(self):
        """More extra macrocycles should produce more iterations."""
        model = LJGaussianModel()

        settings_zero = NT2Settings(
            **{**_SN2_SETTINGS.__dict__, "extra_macrocycles_after_bond_criteria": 0}
        )
        settings_five = NT2Settings(
            **{**_SN2_SETTINGS.__dict__, "extra_macrocycles_after_bond_criteria": 5}
        )

        state0 = _make_sn2_state()
        _, _, vals_zero = nt2_optimize(model, state0, [0, 1], [1, 5], settings_zero)

        state5 = _make_sn2_state()
        _, _, vals_five = nt2_optimize(model, state5, [0, 1], [1, 5], settings_five)

        assert len(vals_five) >= len(vals_zero) + 5, (
            f"5 extra macrocycles should add >= 5 iterations: "
            f"{len(vals_zero)} vs {len(vals_five)}"
        )

    def test_sn2_gradient_consistency(self):
        """Verify LJGaussianModel gradients via finite differences."""
        model = LJGaussianModel()
        state = _make_sn2_state()

        output = model(state)
        forces = output["forces"]

        delta = 1e-5
        pos = state.positions.clone()
        for i in range(min(3, pos.shape[0])):
            for d in range(3):
                pos_plus = pos.clone()
                pos_plus[i, d] += delta
                pos_minus = pos.clone()
                pos_minus[i, d] -= delta

                s_plus = SimState(
                    positions=pos_plus,
                    masses=state.masses.clone(),
                    cell=state.cell.clone(),
                    pbc=state.pbc,
                    atomic_numbers=state.atomic_numbers.clone(),
                )
                s_minus = SimState(
                    positions=pos_minus,
                    masses=state.masses.clone(),
                    cell=state.cell.clone(),
                    pbc=state.pbc,
                    atomic_numbers=state.atomic_numbers.clone(),
                )
                e_plus = model(s_plus)["energy"].item()
                e_minus = model(s_minus)["energy"].item()
                fd_force = -(e_plus - e_minus) / (2 * delta)

                assert abs(forces[i, d].item() - fd_force) < 1e-5, (
                    f"Force mismatch at atom {i}, dim {d}: "
                    f"analytical={forces[i, d].item():.8f}, fd={fd_force:.8f}"
                )


# ===================================================================
# Tests: batched diatomic
# ===================================================================

class TestNT2Batched:
    def test_batch_produces_per_system_results(self):
        model = DiatomicMockModel()
        state = _make_batched_diatomic_state([11.0, 10.5])

        ts_state, trajs, vals = batch_nt2_optimize(
            model, state,
            association_lists=[[0, 1], [0, 1]],
            dissociation_lists=[[], []],
            settings_list=[_ASSOC_SETTINGS, _ASSOC_SETTINGS],
        )

        assert ts_state.n_systems == 2
        assert len(trajs) == 2
        assert len(vals) == 2
        for i in range(2):
            assert len(trajs[i]) > 1
            assert len(vals[i]) == len(trajs[i])

    def test_batch_ts_guess_within_bounds(self):
        model = DiatomicMockModel()
        state = _make_batched_diatomic_state([11.0, 10.5])

        ts_state, _, _ = batch_nt2_optimize(
            model, state,
            association_lists=[[0, 1], [0, 1]],
            dissociation_lists=[[], []],
            settings_list=[_ASSOC_SETTINGS, _ASSOC_SETTINGS],
        )

        for s in range(2):
            mask = ts_state.system_idx == s
            pos = ts_state.positions[mask]
            r_ts = (pos[0] - pos[1]).norm().item()
            assert 5.0 < r_ts < 11.5, f"System {s}: TS distance {r_ts} out of range"

    def test_batch_matches_single(self):
        """Batched run with 1 system should match single-system run."""
        model = DiatomicMockModel()
        r = 11.0

        state_single = _make_diatomic_state(r=r)
        ts_single, traj_single, vals_single = nt2_optimize(
            model, state_single, [0, 1], [], _ASSOC_SETTINGS,
        )

        state_batch = _make_diatomic_state(r=r)
        ts_batch, trajs_batch, vals_batch = batch_nt2_optimize(
            model, state_batch,
            association_lists=[[0, 1]],
            dissociation_lists=[[]],
            settings_list=[_ASSOC_SETTINGS],
        )

        assert len(vals_single) == len(vals_batch[0])
        for v1, v2 in zip(vals_single, vals_batch[0]):
            assert abs(v1 - v2) < 1e-10, f"Energy mismatch: {v1} vs {v2}"

        assert torch.allclose(ts_single.positions, ts_batch.positions, atol=1e-10)

    def test_batch_different_settings(self):
        """Systems with different max_iter should converge independently."""
        model = DiatomicMockModel()
        state = _make_batched_diatomic_state([11.0, 11.0])

        s_fast = NT2Settings(**{**_ASSOC_SETTINGS.__dict__, "max_iter": 5})
        s_slow = NT2Settings(**{**_ASSOC_SETTINGS.__dict__, "max_iter": 50})

        _, trajs, _ = batch_nt2_optimize(
            model, state,
            association_lists=[[0, 1], [0, 1]],
            dissociation_lists=[[], []],
            settings_list=[s_fast, s_slow],
        )

        assert len(trajs[0]) <= 5
        assert len(trajs[1]) <= 50
        assert len(trajs[1]) > len(trajs[0])

    def test_mismatched_list_lengths_raises(self):
        model = DiatomicMockModel()
        state = _make_batched_diatomic_state([11.0, 10.5])
        with pytest.raises(ValueError, match="association lists"):
            batch_nt2_optimize(model, state, [[0, 1]], [[], []])


# ===================================================================
# Tests: batched SN2
# ===================================================================

class TestNT2BatchedSN2:
    """SN2 tests in batched mode with the LJGaussianModel."""

    def test_batched_sn2_qualitative(self):
        """Two copies of SN2 should both show association/dissociation."""
        model = LJGaussianModel()
        state = _make_batched_sn2_state(2)

        ts_state, trajs, vals = batch_nt2_optimize(
            model, state,
            association_lists=[[0, 1], [0, 1]],
            dissociation_lists=[[1, 5], [1, 5]],
            settings_list=[_SN2_SETTINGS, _SN2_SETTINGS],
        )

        assert ts_state.n_systems == 2
        for s in range(2):
            assert len(trajs[s]) > 1

    def test_batched_sn2_matches_single(self):
        """Batched SN2 with 1 copy should match single run."""
        model = LJGaussianModel()

        state_single = _make_sn2_state()
        ts_single, _, vals_single = nt2_optimize(
            model, state_single, [0, 1], [1, 5], _SN2_SETTINGS,
        )

        state_batch = _make_sn2_state()
        ts_batch, _, vals_batch = batch_nt2_optimize(
            model, state_batch,
            association_lists=[[0, 1]],
            dissociation_lists=[[1, 5]],
            settings_list=[_SN2_SETTINGS],
        )

        assert len(vals_single) == len(vals_batch[0])
        for v1, v2 in zip(vals_single, vals_batch[0]):
            assert abs(v1 - v2) < 1e-10
        assert torch.allclose(ts_single.positions, ts_batch.positions, atol=1e-10)
