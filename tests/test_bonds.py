"""Tests for distance-based bond detection."""

import torch
import pytest

from torch_sim.bonds import detect_bonds


class TestDetectBonds:
    def test_h2_bonded(self):
        """Two H atoms at 0.74 A should be bonded."""
        atomic_numbers = torch.tensor([1, 1])
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=torch.float64)
        bonds = detect_bonds(atomic_numbers, positions)
        assert bonds.shape == (2, 2)
        assert bonds[0, 1] == 1.0
        assert bonds[1, 0] == 1.0
        assert bonds[0, 0] == 0.0

    def test_h2_too_far(self):
        """Two H atoms at 5 A should not be bonded."""
        atomic_numbers = torch.tensor([1, 1])
        positions = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float64)
        bonds = detect_bonds(atomic_numbers, positions)
        assert bonds[0, 1] == 0.0

    def test_water_bonds(self):
        """Water molecule: O bonded to both H, H's not bonded to each other."""
        atomic_numbers = torch.tensor([8, 1, 1])
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ], dtype=torch.float64)
        bonds = detect_bonds(atomic_numbers, positions)
        assert bonds[0, 1] == 1.0  # O-H bond
        assert bonds[0, 2] == 1.0  # O-H bond
        assert bonds[1, 2] == 0.0  # H-H no bond

    def test_symmetry(self):
        atomic_numbers = torch.tensor([6, 8, 1, 1])
        positions = torch.randn(4, 3, dtype=torch.float64) * 0.5
        bonds = detect_bonds(atomic_numbers, positions)
        assert torch.allclose(bonds, bonds.T)

    def test_diagonal_zero(self):
        atomic_numbers = torch.tensor([6, 6, 6])
        positions = torch.zeros(3, 3, dtype=torch.float64)
        bonds = detect_bonds(atomic_numbers, positions)
        assert bonds.diag().sum() == 0.0

    def test_ch4_bonds(self):
        """Methane: C bonded to 4 H, no H-H bonds."""
        atomic_numbers = torch.tensor([6, 1, 1, 1, 1])
        positions = torch.tensor([
            [0.000, 0.000, 0.000],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ], dtype=torch.float64)
        bonds = detect_bonds(atomic_numbers, positions)
        for h_idx in range(1, 5):
            assert bonds[0, h_idx] == 1.0, f"C-H{h_idx} should be bonded"
        for i in range(1, 5):
            for j in range(i + 1, 5):
                assert bonds[i, j] == 0.0, f"H{i}-H{j} should not be bonded"
