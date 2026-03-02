"""Dimer optimizer for Hessian-free saddle point (transition state) search.

Reimplements the SCINE Dimer algorithm: constructs a dimer on the PES,
iteratively rotates it to align with the lowest curvature mode, then
translates toward the saddle point by inverting the gradient component
along the dimer axis.

All default values are in Angstrom / eV units (converted from SCINE's
Bohr / Hartree defaults via 1 Bohr = 0.529177 A, 1 Ha/Bohr = 51.422 eV/A).

Based on:
    Kaestner J. and Sherwood P., J. Chem. Phys., 2008, DOI: 10.1063/1.2815812
    Shang C. and Liu ZP., J. Chem. Theory Comput., 2010, DOI: 10.1021/ct9005147
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from torch_sim.optimizers.irc import gradient_based_converged
from torch_sim.state import SimState

if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface

logger = logging.getLogger(__name__)

# SCINE uses Bohr / Hartree.  torch-sim uses Angstrom / eV.
_BOHR_TO_ANG = 0.529177
_HA_BOHR_TO_EV_ANG = 51.422  # 1 Hartree/Bohr in eV/Angstrom
_HA_BOHR2_TO_EV_ANG2 = 97.17  # 1 Hartree/Bohr^2 in eV/Angstrom^2


@dataclass
class DimerSettings:
    """Settings for the Dimer optimizer (Angstrom / eV units)."""

    radius: float = 0.01 * _BOHR_TO_ANG  # ~0.00529 A
    trust_radius: float = 0.2 * _BOHR_TO_ANG  # ~0.106 A
    default_translation_step: float = 1.0  # dimensionless
    max_rotations_first_cycle: int = 100
    max_rotations_other_cycles: int = 100
    interval_of_rotations: int = 5
    phi_tolerance: float = 1e-3  # radians
    # dCdphi threshold (curvature derivative, eV/A^2)
    rotation_gradient_threshold_first_cycle: float = 1e-7 * _HA_BOHR2_TO_EV_ANG2
    # ortho_g norm threshold (eV/A)
    rotation_gradient_threshold_other_cycles: float = 1e-4 * _HA_BOHR_TO_EV_ANG
    lowered_rotation_gradient_threshold: float = 1e-3 * _HA_BOHR_TO_EV_ANG
    decrease_rotation_gradient_threshold: bool = False
    cycle_of_rotation_gradient_decrease: int = 5
    gradient_interpolation: bool = True
    rotation_lbfgs: bool = True
    rotation_cg: bool = False
    only_one_rotation: bool = False
    skip_first_rotation: bool = False
    lbfgs_memory: int = 5
    bfgs_start: int = 16
    minimization_cycle: int = 5
    multi_scale: bool = True
    grad_rmsd_threshold: float = 1e-3 * _HA_BOHR_TO_EV_ANG  # ~0.051 eV/A
    rotation_force_threshold: float = 1e-3 * _HA_BOHR_TO_EV_ANG  # ~0.051 eV/A
    max_iter: int = 500
    max_value_memory: int = 10
    step_max_coeff: float = 2.0e-3
    step_rms: float = 1.0e-3
    grad_max_coeff: float = 2.0e-4
    grad_rms: float = 1.0e-4
    delta_value: float = 1.0e-6
    convergence_requirement: int = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_oscillating(mem: deque) -> bool:
    """Check if stored values show an alternating sign pattern."""
    n = len(mem)
    if n < 3:
        return False
    if abs(mem[0] - mem[1]) < 1e-12:
        return False
    prev_positive = (mem[0] - mem[1]) > 0.0
    for i in range(2, n):
        this_positive = (mem[i - 1] - mem[i]) > 0.0
        if prev_positive == this_positive:
            return False
        prev_positive = this_positive
    return True


def _gradient_below_threshold(
    gradients: torch.Tensor, threshold: float
) -> bool:
    n = gradients.shape[0]
    return float(math.sqrt(gradients.dot(gradients).item() / n)) < threshold


def _cg_rotation(
    ortho_fn: torch.Tensor,
    prev_ortho_fn: torch.Tensor,
    prev_ortho_g: torch.Tensor,
    rotation_cycle: int,
) -> torch.Tensor:
    """Polak-Ribiere conjugate gradient for dimer rotation."""
    ortho_g = ortho_fn.clone()
    if rotation_cycle != 0:
        denom = prev_ortho_fn.dot(prev_ortho_fn)
        if abs(denom.item()) > 1e-20:
            beta = ortho_fn.dot(ortho_fn - prev_ortho_fn) / denom
            ortho_g = ortho_fn + beta * prev_ortho_g
    return ortho_g


def _lbfgs_rotation(
    ortho_fn: torch.Tensor,
    prev_ortho_fn: torch.Tensor,
    params_r1: torch.Tensor,
    old_params_r1: torch.Tensor,
    dx_mat: torch.Tensor,
    dg_mat: torch.Tensor,
    rotation_cycle: int,
    m_counter: int,
    max_m: int,
) -> tuple[torch.Tensor, int]:
    """L-BFGS two-loop recursion for dimer rotation."""
    ortho_g = ortho_fn.clone()
    if rotation_cycle == 0:
        dx_mat.zero_()
        dg_mat.zero_()
        return ortho_g, m_counter

    if m_counter < max_m:
        dg_mat[:, m_counter] = prev_ortho_fn - ortho_fn
        dx_mat[:, m_counter] = params_r1 - old_params_r1
        m_counter += 1
    else:
        dg_mat[:, :-1] = dg_mat[:, 1:].clone()
        dx_mat[:, :-1] = dx_mat[:, 1:].clone()
        dg_mat[:, max_m - 1] = prev_ortho_fn - ortho_fn
        dx_mat[:, max_m - 1] = params_r1 - old_params_r1

    m = m_counter
    alpha_vec = torch.zeros(m, device=ortho_fn.device, dtype=ortho_fn.dtype)
    for i in range(m - 1, -1, -1):
        dx_dot_dg = dx_mat[:, i].dot(dg_mat[:, i])
        if abs(dx_dot_dg.item()) < 1e-6:
            sign = -1e-6 if dx_dot_dg.item() < 0 else 1e-6
            alpha_vec[i] = dx_mat[:, i].dot(ortho_g) / sign
        else:
            alpha_vec[i] = dx_mat[:, i].dot(ortho_g) / dx_dot_dg
        ortho_g = ortho_g - alpha_vec[i] * dg_mat[:, i]

    scale = dx_mat[:, m - 1].dot(dg_mat[:, m - 1]) / dg_mat[:, m - 1].dot(
        dg_mat[:, m - 1]
    )
    ortho_g = ortho_g * scale

    for i in range(m):
        dx_dot_dg = dx_mat[:, i].dot(dg_mat[:, i])
        beta_val = dg_mat[:, i].dot(ortho_g)
        if abs(dx_dot_dg.item()) < 1e-6:
            beta_val = beta_val / 1e-6
        else:
            beta_val = beta_val / dx_dot_dg
        ortho_g = ortho_g + (alpha_vec[i] - beta_val) * dx_mat[:, i]

    return ortho_g, m_counter


def _determine_direction(
    value: float,
    parameters: torch.Tensor,
    value_r1: float,
    gradients: torch.Tensor,
    gradients_r1: torch.Tensor,
    dimer_axis: torch.Tensor,
    radius: float,
    eval_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Flip dimer axis so it points toward higher energy."""
    if value <= value_r1:
        curv = ((gradients_r1 - gradients).dot(dimer_axis) / radius).item()
        return dimer_axis, gradients_r1, parameters + radius * dimer_axis, value_r1, curv

    curvature_old = ((gradients_r1 - gradients).dot(dimer_axis) / radius).item()
    dimer_axis = -dimer_axis
    params_r1_new = parameters + dimer_axis * radius
    value_r1_new, grads_r1_new = eval_fn(params_r1_new)

    if value > value_r1_new:
        curvature_new = ((grads_r1_new - gradients).dot(dimer_axis) / radius).item()
        if curvature_new < curvature_old:
            return dimer_axis, grads_r1_new, params_r1_new, value_r1_new, curvature_new
        dimer_axis = -dimer_axis
        return dimer_axis, gradients_r1, parameters + radius * dimer_axis, value_r1, curvature_old

    curv = ((grads_r1_new - gradients).dot(dimer_axis) / radius).item()
    return dimer_axis, grads_r1_new, params_r1_new, value_r1_new, curv


# ---------------------------------------------------------------------------
# Single-system core optimizer
# ---------------------------------------------------------------------------


def dimer_optimize(
    parameters: torch.Tensor,
    eval_fn: Callable[[torch.Tensor], tuple[float, torch.Tensor]],
    settings: DimerSettings | None = None,
    guess_vector: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Run the Dimer algorithm to find a saddle point.

    Args:
        parameters: Initial parameter vector of shape ``[n_params]``.
        eval_fn: Callable ``(params) -> (energy, gradients)``.
        settings: DimerSettings (defaults used if None).
        guess_vector: Optional initial dimer axis vector of shape ``[n_params]``.

    Returns:
        Tuple of ``(optimized_parameters, n_cycles)``.
    """
    if settings is None:
        settings = DimerSettings()

    n_params = parameters.shape[0]
    if n_params == 0:
        raise ValueError("Empty parameter vector")

    device = parameters.device
    dtype = parameters.dtype

    params = parameters.clone()
    default_step = settings.default_translation_step
    current_step_length = default_step

    # State vectors
    gradients = torch.zeros(n_params, device=device, dtype=dtype)
    gradients_r1 = torch.zeros(n_params, device=device, dtype=dtype)
    old_gradients = torch.zeros(n_params, device=device, dtype=dtype)
    old_parameters = torch.zeros(n_params, device=device, dtype=dtype)
    params_r1 = torch.zeros(n_params, device=device, dtype=dtype)
    old_params_r1 = torch.zeros(n_params, device=device, dtype=dtype)
    f_para = torch.zeros(n_params, device=device, dtype=dtype)
    f_para_old = torch.zeros(n_params, device=device, dtype=dtype)
    mod_gradient = torch.zeros(n_params, device=device, dtype=dtype)
    stepvector = torch.zeros(n_params, device=device, dtype=dtype)
    steps = torch.zeros(n_params, device=device, dtype=dtype)
    ortho_fn = torch.zeros(n_params, device=device, dtype=dtype)
    prev_ortho_fn = torch.zeros(n_params, device=device, dtype=dtype)
    ortho_g = torch.zeros(n_params, device=device, dtype=dtype)
    prev_ortho_g = torch.zeros(n_params, device=device, dtype=dtype)
    dx_mat = torch.zeros(n_params, settings.lbfgs_memory, device=device, dtype=dtype)
    dg_mat = torch.zeros(n_params, settings.lbfgs_memory, device=device, dtype=dtype)
    inv_h = 0.5 * torch.eye(n_params, device=device, dtype=dtype)

    curvature = 0.0
    applied_stepsize_scaling = False
    n_rotation_cycles_performed = 0
    interval_of_rotations = settings.interval_of_rotations
    value_memory: deque[float] = deque(maxlen=settings.max_value_memory)
    check_old_params = params.clone()

    # --- First evaluation ---
    value, gradients = eval_fn(params)
    check_old_value = value

    # --- Create dimer axis ---
    if guess_vector is not None and guess_vector.shape[0] == n_params:
        dimer_axis = guess_vector.clone().to(device=device, dtype=dtype)
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        dimer_axis = torch.randn(n_params, device=device, dtype=dtype, generator=gen)
    dimer_axis = dimer_axis / dimer_axis.norm()

    params_r1 = params + settings.radius * dimer_axis
    value_r1, gradients_r1 = eval_fn(params_r1)
    dimer_axis, gradients_r1, params_r1, value_r1, curvature = _determine_direction(
        value, params, value_r1, gradients, gradients_r1,
        dimer_axis, settings.radius, eval_fn,
    )

    # --- Main loop ---
    for cycle in range(1, settings.max_iter + 1):
        params_r1 = params + settings.radius * dimer_axis
        f_para_old = f_para.clone()
        f_para = -gradients.dot(dimer_axis) * dimer_axis

        max_rot_cycles = (
            settings.max_rotations_first_cycle if cycle == 1
            else settings.max_rotations_other_cycles
        )

        # --- Rotation decision ---
        perform_rotation = False
        if cycle == 1:
            if not settings.skip_first_rotation:
                _, gradients_r1 = eval_fn(params_r1)
                ortho_fn = (
                    2.0 * (gradients_r1 - gradients).dot(dimer_axis) * dimer_axis
                    - 2.0 * (gradients_r1 - gradients)
                )
                curvature = ((gradients_r1 - gradients).dot(dimer_axis) / settings.radius).item()
                perform_rotation = True
        elif not settings.only_one_rotation:
            if (cycle % interval_of_rotations) == 0:
                _, gradients_r1 = eval_fn(params_r1)
                curvature = ((gradients_r1 - gradients).dot(dimer_axis) / settings.radius).item()
                ortho_fn = (
                    2.0 * (gradients_r1 - gradients).dot(dimer_axis) * dimer_axis
                    - 2.0 * (gradients_r1 - gradients)
                )
                if (curvature > 0 and f_para.norm() < f_para_old.norm()) or (
                    curvature < 0 and f_para.norm() > f_para_old.norm()
                ):
                    perform_rotation = True
                elif ortho_fn.norm().item() > settings.rotation_force_threshold:
                    perform_rotation = True

        if perform_rotation:
            n_rotation_cycles_performed += 1
            if cycle == 1:
                # --- rotationWithPhi ---
                reached_tol = False
                m_counter = 0
                phi_min = 0.0
                phi1 = 0.0
                grads_r1_phi1 = torch.zeros_like(gradients)

                for rc in range(max_rot_cycles):
                    if settings.gradient_interpolation and rc != 0:
                        gradients_r1 = (
                            math.sin(phi1 - phi_min) / math.sin(phi1) * gradients_r1
                            + math.sin(phi_min) / math.sin(phi1) * grads_r1_phi1
                            + (1 - math.cos(phi_min) - math.sin(phi_min) * math.tan(phi1 / 2)) * gradients
                        )
                    else:
                        _, gradients_r1 = eval_fn(params_r1)

                    ortho_dimer = dimer_axis.clone()
                    ortho_fn = (
                        2.0 * (gradients_r1 - gradients).dot(dimer_axis) * dimer_axis
                        - 2.0 * (gradients_r1 - gradients)
                    )
                    ortho_g = ortho_fn.clone()
                    if settings.rotation_cg:
                        ortho_g = _cg_rotation(ortho_fn, prev_ortho_fn, prev_ortho_g, rc)
                    elif settings.rotation_lbfgs:
                        ortho_g, m_counter = _lbfgs_rotation(
                            ortho_fn, prev_ortho_fn, params_r1, old_params_r1,
                            dx_mat, dg_mat, rc, m_counter, settings.lbfgs_memory,
                        )
                    ortho_g = ortho_g - ortho_g.dot(dimer_axis) * dimer_axis
                    theta_norm = ortho_g.norm()
                    if theta_norm < 1e-20:
                        reached_tol = True
                        break
                    theta = ortho_g / theta_norm

                    prev_ortho_g = ortho_g.clone()
                    prev_ortho_fn = ortho_fn.clone()
                    old_params_r1 = params_r1.clone()

                    c0 = ((gradients_r1 - gradients).dot(dimer_axis) / settings.radius).item()
                    dc_dphi = (2.0 * (gradients_r1 - gradients).dot(theta) / settings.radius).item()

                    if abs(dc_dphi) < settings.rotation_gradient_threshold_first_cycle:
                        reached_tol = True
                        if rc == 0:
                            interval_of_rotations += 1
                        break

                    phi1 = -0.5 * math.atan2(dc_dphi, 2.0 * abs(c0))
                    if abs(phi1) < settings.phi_tolerance:
                        reached_tol = True
                        break

                    dimer_axis = ortho_dimer * math.cos(phi1) + theta * math.sin(phi1)
                    dimer_axis = dimer_axis / dimer_axis.norm()
                    params_r1 = params + settings.radius * dimer_axis
                    _, grads_r1_phi1 = eval_fn(params_r1)
                    c_phi1 = ((grads_r1_phi1 - gradients).dot(dimer_axis) / settings.radius).item()

                    b1 = 0.5 * dc_dphi
                    denom_fourier = 1 - math.cos(2 * phi1)
                    if abs(denom_fourier) < 1e-20:
                        denom_fourier = 1e-20
                    a1 = (c0 - c_phi1 + b1 * math.sin(2 * phi1)) / denom_fourier
                    a0 = 2 * (c0 - a1)
                    phi_min = 0.5 * math.atan2(b1, a1)
                    c_phi_min = a0 / 2.0 + a1 * math.cos(2.0 * phi_min) + b1 * math.sin(2.0 * phi_min)

                    if c_phi_min > c0 and c_phi_min > c_phi1:
                        phi_min += math.pi / 2
                        curvature = a0 / 2.0 + a1 * math.cos(2.0 * phi_min) + b1 * math.sin(2.0 * phi_min)
                    elif c_phi1 < c_phi_min:
                        phi_min = phi1
                        curvature = c_phi1
                    elif c0 < c_phi_min:
                        phi_min = 0.0
                        curvature = c0

                    dimer_axis = ortho_dimer
                    params_r1 = params + settings.radius * dimer_axis
                    if abs(phi_min) < settings.phi_tolerance:
                        reached_tol = True
                        break
                    dimer_axis = ortho_dimer * math.cos(phi_min) + theta * math.sin(phi_min)
                    dimer_axis = dimer_axis / dimer_axis.norm()
                    params_r1 = params + settings.radius * dimer_axis

                if not reached_tol:
                    _, gradients_r1 = eval_fn(params_r1)
                dimer_axis, gradients_r1, params_r1, value_r1, curvature = _determine_direction(
                    value, params, value_r1, gradients, gradients_r1,
                    dimer_axis, settings.radius, eval_fn,
                )
            else:
                # --- rotationWithGradient ---
                pre_rotation_axis = dimer_axis.clone()
                rot_thresh = settings.rotation_gradient_threshold_other_cycles
                m_counter = 0
                if (settings.decrease_rotation_gradient_threshold
                        and n_rotation_cycles_performed > settings.cycle_of_rotation_gradient_decrease):
                    rot_thresh = settings.lowered_rotation_gradient_threshold

                for rc in range(max_rot_cycles):
                    ortho_fn = (
                        2.0 * (gradients_r1 - gradients).dot(dimer_axis) * dimer_axis
                        - 2.0 * (gradients_r1 - gradients)
                    )
                    ortho_g = ortho_fn.clone()
                    if settings.rotation_cg:
                        ortho_g = _cg_rotation(ortho_fn, prev_ortho_fn, prev_ortho_g, rc)
                    elif settings.rotation_lbfgs:
                        ortho_g, m_counter = _lbfgs_rotation(
                            ortho_fn, prev_ortho_fn, params_r1, old_params_r1,
                            dx_mat, dg_mat, rc, m_counter, settings.lbfgs_memory,
                        )
                    prev_ortho_g = ortho_g.clone()
                    prev_ortho_fn = ortho_fn.clone()
                    old_params_r1 = params_r1.clone()
                    params_r1 = params_r1 + ortho_g
                    dimer_axis = params_r1 - params
                    dimer_axis = dimer_axis / dimer_axis.norm()
                    params_r1 = params + dimer_axis * settings.radius
                    _, gradients_r1 = eval_fn(params_r1)
                    curvature = ((gradients_r1 - gradients).dot(dimer_axis) / settings.radius).item()
                    if ortho_g.norm().item() < rot_thresh:
                        cos_sim = dimer_axis.dot(pre_rotation_axis).item()
                        if cos_sim > 0.99:
                            interval_of_rotations += 1
                        dimer_axis, gradients_r1, params_r1, value_r1, curvature = _determine_direction(
                            value, params, value_r1, gradients, gradients_r1,
                            dimer_axis, settings.radius, eval_fn,
                        )
                        break
                else:
                    interval_of_rotations = 1
                    dimer_axis, gradients_r1, params_r1, value_r1, curvature = _determine_direction(
                        value, params, value_r1, gradients, gradients_r1,
                        dimer_axis, settings.radius, eval_fn,
                    )

            current_step_length = default_step

        # --- Modified gradient ---
        if curvature > 0 and cycle < settings.minimization_cycle:
            mod_gradient = -gradients.dot(dimer_axis) * dimer_axis
        else:
            mod_gradient = -2.0 * (gradients.dot(dimer_axis) * dimer_axis) + gradients

        # --- BFGS translation ---
        if cycle == 1:
            old_parameters = params.clone()
            old_gradients = mod_gradient.clone()
            stepvector = -mod_gradient
        else:
            dx = params - old_parameters
            dg = mod_gradient - old_gradients
            dx_dot_dg = dx.dot(dg).item()
            dg_t_inv_h_dg = (dg @ inv_h @ dg).item()

            half_eye = 0.5 * torch.eye(n_params, device=device, dtype=dtype)
            if torch.allclose(inv_h, half_eye, atol=1e-10):
                dg_dot_dg = dg.dot(dg).item()
                if abs(dg_dot_dg) > 1e-20:
                    inv_h.diagonal().mul_(2.0 * dx_dot_dg / dg_dot_dg)

            sigma2, sigma3 = 0.9, 9.0
            delta_pw = 1.0
            if abs(dx_dot_dg) < abs((1.0 - sigma2) * dg_t_inv_h_dg):
                delta_pw = sigma2 * dg_t_inv_h_dg / (dg_t_inv_h_dg - dx_dot_dg)
            elif abs(dx_dot_dg) > abs((1.0 + sigma3) * dg_t_inv_h_dg):
                delta_pw = -sigma3 * dg_t_inv_h_dg / (dg_t_inv_h_dg - dx_dot_dg)
            if abs(delta_pw - 1.0) >= 1e-16:
                dx = delta_pw * dx + (1.0 - delta_pw) * (inv_h @ dg)
                dx_dot_dg = dx.dot(dg).item()
            if abs(dx_dot_dg) < 1e-9:
                dx_dot_dg = -1e-9 if dx_dot_dg < 0 else 1e-9

            alpha_bfgs = (dx_dot_dg + (dg @ inv_h @ dg).item()) / (dx_dot_dg ** 2)
            beta_bfgs = 1.0 / dx_dot_dg
            inv_h = (
                inv_h + alpha_bfgs * torch.outer(dx, dx)
                - beta_bfgs * (inv_h @ torch.outer(dg, dx) + torch.outer(dx, dg) @ inv_h)
            )
            old_parameters = params.clone()
            old_gradients = mod_gradient.clone()

            if (not applied_stepsize_scaling
                    and (cycle >= settings.bfgs_start
                         or _gradient_below_threshold(gradients, settings.grad_rmsd_threshold))):
                applied_stepsize_scaling = True
            if applied_stepsize_scaling:
                stepvector = -(inv_h @ mod_gradient)
            else:
                stepvector = -mod_gradient

        steps = default_step * stepvector

        # Trust radius
        max_val = steps.abs().max().item()
        if max_val > settings.trust_radius:
            steps = steps * (settings.trust_radius / max_val)
            inv_h = 0.5 * torch.eye(n_params, device=device, dtype=dtype)
            applied_stepsize_scaling = False
            current_step_length = default_step

        params = params + steps
        value, gradients = eval_fn(params)

        # Convergence
        delta_param = params - check_old_params
        delta_v = value - check_old_value
        check_old_params = params.clone()
        check_old_value = value

        stop = cycle >= settings.max_iter or gradient_based_converged(
            gradients, delta_param, delta_v,
            settings.grad_max_coeff, settings.grad_rms,
            settings.step_max_coeff, settings.step_rms,
            settings.delta_value, settings.convergence_requirement,
        )
        if stop:
            return params, cycle

        # Oscillation
        value_memory.append(value)
        if _is_oscillating(value_memory):
            params = params - steps / 2.0
            value, gradients = eval_fn(params)
            default_step *= 0.95
            current_step_length = default_step
        else:
            default_step = 1.0
            current_step_length = default_step

    return params, settings.max_iter


# ---------------------------------------------------------------------------
# SimState / ModelInterface wrappers
# ---------------------------------------------------------------------------


def dimer_ts_optimize(
    model: ModelInterface,
    state: SimState,
    settings: DimerSettings | None = None,
    guess_vector: torch.Tensor | None = None,
) -> tuple[SimState, int]:
    """Run Dimer saddle-point optimization on a single-system SimState.

    Args:
        model: ModelInterface returning energy and forces.
        state: Single-system SimState (initial guess for the TS).
        settings: DimerSettings (defaults used if None).
        guess_vector: Optional initial dimer axis direction ``[3*n_atoms]``.

    Returns:
        Tuple of ``(optimized SimState, number of cycles)``.
    """
    if state.n_systems != 1:
        raise ValueError(
            "dimer_ts_optimize expects a single-system SimState. "
            "Use batch_dimer_ts_optimize for multi-system batching."
        )

    n_atoms = state.n_atoms

    def _eval_fn(params: torch.Tensor) -> tuple[float, torch.Tensor]:
        pos = params.reshape(n_atoms, 3)
        s = SimState(
            positions=pos, masses=state.masses, cell=state.cell,
            pbc=state.pbc, atomic_numbers=state.atomic_numbers,
        )
        out = model(s)
        return float(out["energy"].item()), -out["forces"].detach().reshape(-1)

    params = state.positions.detach().clone().reshape(-1)
    opt_params, n_cycles = dimer_optimize(params, _eval_fn, settings, guess_vector)

    return SimState(
        positions=opt_params.reshape(n_atoms, 3),
        masses=state.masses.clone(), cell=state.cell.clone(),
        pbc=state.pbc, atomic_numbers=state.atomic_numbers.clone(),
    ), n_cycles


# ---------------------------------------------------------------------------
# Batched evaluation helper
# ---------------------------------------------------------------------------


def _eval_systems(
    model: ModelInterface,
    ref_state: SimState,
    sys_indices: list[int],
    per_sys_positions: list[torch.Tensor],
    counts: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[dict[int, float], dict[int, torch.Tensor]]:
    """Evaluate model for specified systems at given positions (batched).

    Args:
        model: ModelInterface.
        ref_state: Original batched SimState for masses, cell, atomic_numbers.
        sys_indices: Which systems to evaluate.
        per_sys_positions: List of ``[n_atoms_s, 3]`` tensors, one per sys_index.
        counts: Per-system atom counts from ref_state.
        offsets: Cumulative atom offsets.

    Returns:
        (energies_dict, gradients_dict) mapping sys_idx -> value.
    """
    if not sys_indices:
        return {}, {}

    all_pos, all_masses, all_z, all_cells, all_sidx = [], [], [], [], []
    for i, s in enumerate(sys_indices):
        n_at = int(counts[s].item())
        off = int(offsets[s].item())
        all_pos.append(per_sys_positions[i])
        all_masses.append(ref_state.masses[off: off + n_at])
        all_z.append(ref_state.atomic_numbers[off: off + n_at])
        all_cells.append(ref_state.cell[s: s + 1])
        all_sidx.append(torch.full((n_at,), i, dtype=torch.long, device=ref_state.device))

    eval_state = SimState(
        positions=torch.cat(all_pos),
        masses=torch.cat(all_masses),
        cell=torch.cat(all_cells),
        pbc=ref_state.pbc,
        atomic_numbers=torch.cat(all_z),
        system_idx=torch.cat(all_sidx),
    )
    out = model(eval_state)

    energies, gradients = {}, {}
    off = 0
    for i, s in enumerate(sys_indices):
        n_at = int(counts[s].item())
        energies[s] = float(out["energy"][i].item())
        gradients[s] = -out["forces"][off: off + n_at].reshape(-1)
        off += n_at
    return energies, gradients


# ---------------------------------------------------------------------------
# Batched Dimer optimizer (model calls are batched across systems)
# ---------------------------------------------------------------------------


def batch_dimer_ts_optimize(
    model: ModelInterface,
    state: SimState,
    settings_list: list[DimerSettings | None] | None = None,
    guess_vectors: list[torch.Tensor | None] | None = None,
) -> tuple[SimState, list[int]]:
    """Run Dimer saddle-point optimization on a batched SimState.

    Model evaluations are batched across systems for GPU efficiency.
    Per-system state (dimer axes, BFGS inverse Hessians, etc.) is
    maintained independently.

    Args:
        model: ModelInterface returning energy and forces (batched).
        state: Batched SimState with ``n_systems`` systems.
        settings_list: Per-system DimerSettings.  If *None*, defaults
            are used for every system.
        guess_vectors: Per-system initial dimer axis vectors.  If *None*,
            random axes are used.

    Returns:
        Tuple of ``(optimized SimState, list of per-system cycle counts)``.
    """
    n_systems = state.n_systems
    if settings_list is None:
        settings_list = [None] * n_systems
    settings_list = [s or DimerSettings() for s in settings_list]
    if guess_vectors is None:
        guess_vectors = [None] * n_systems

    device = state.device
    dtype = state.positions.dtype
    system_idx = state.system_idx
    counts = torch.bincount(system_idx, minlength=n_systems)
    offsets = torch.zeros(n_systems + 1, device=device, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)
    n_params = [int(c.item()) * 3 for c in counts]

    def _extract_pos(s: int) -> torch.Tensor:
        o, c = int(offsets[s].item()), int(counts[s].item())
        return state.positions[o: o + c].clone()

    def _eval_all(positions: list[torch.Tensor]) -> tuple[dict, dict]:
        return _eval_systems(
            model, state, list(range(n_systems)), positions, counts, offsets,
        )

    def _eval_subset(indices: list[int], positions: list[torch.Tensor]) -> tuple[dict, dict]:
        return _eval_systems(model, state, indices, positions, counts, offsets)

    # --- Per-system state ---
    params = [_extract_pos(s).reshape(-1) for s in range(n_systems)]
    dimer_axis = [None] * n_systems
    grads = [None] * n_systems
    grads_r1 = [None] * n_systems
    old_params = [None] * n_systems
    old_grads = [None] * n_systems
    f_para = [torch.zeros(n_params[s], device=device, dtype=dtype) for s in range(n_systems)]
    f_para_old = [torch.zeros(n_params[s], device=device, dtype=dtype) for s in range(n_systems)]
    inv_h = [0.5 * torch.eye(n_params[s], device=device, dtype=dtype) for s in range(n_systems)]
    dx_mat = [torch.zeros(n_params[s], settings_list[s].lbfgs_memory, device=device, dtype=dtype) for s in range(n_systems)]
    dg_mat = [torch.zeros(n_params[s], settings_list[s].lbfgs_memory, device=device, dtype=dtype) for s in range(n_systems)]
    prev_ortho_fn = [torch.zeros(n_params[s], device=device, dtype=dtype) for s in range(n_systems)]
    prev_ortho_g = [torch.zeros(n_params[s], device=device, dtype=dtype) for s in range(n_systems)]

    curvature = [0.0] * n_systems
    applied_stepsize_scaling = [False] * n_systems
    n_rot_performed = [0] * n_systems
    interval_rot = [settings_list[s].interval_of_rotations for s in range(n_systems)]
    default_step = [settings_list[s].default_translation_step for s in range(n_systems)]
    value_memory = [deque(maxlen=settings_list[s].max_value_memory) for s in range(n_systems)]
    check_old_params = [p.clone() for p in params]
    check_old_value = [0.0] * n_systems
    converged = [False] * n_systems
    cycle_counts = [0] * n_systems

    # --- Initial evaluation (batched) ---
    pos_3d = [p.reshape(-1, 3) for p in params]
    e_dict, g_dict = _eval_all(pos_3d)
    for s in range(n_systems):
        grads[s] = g_dict[s]
        check_old_value[s] = e_dict[s]

    # --- Create dimer axes ---
    for s in range(n_systems):
        if guess_vectors[s] is not None and guess_vectors[s].shape[0] == n_params[s]:
            dimer_axis[s] = guess_vectors[s].clone().to(device=device, dtype=dtype)
        else:
            gen = torch.Generator(device=device)
            gen.manual_seed(42)
            dimer_axis[s] = torch.randn(n_params[s], device=device, dtype=dtype, generator=gen)
        dimer_axis[s] = dimer_axis[s] / dimer_axis[s].norm()

    # Initial R1 evaluation (batched)
    r1_pos = [(params[s] + settings_list[s].radius * dimer_axis[s]).reshape(-1, 3)
              for s in range(n_systems)]
    e_r1, g_r1 = _eval_all(r1_pos)
    val_r1 = [0.0] * n_systems
    for s in range(n_systems):
        grads_r1[s] = g_r1[s]
        val_r1[s] = e_r1[s]

    # Direction determination (may need extra evals)
    for s in range(n_systems):
        def _single_eval(p, _s=s):
            e_d, g_d = _eval_subset([_s], [p.reshape(-1, 3)])
            return e_d[_s], g_d[_s]

        dimer_axis[s], grads_r1[s], _, vr1_new, curvature[s] = _determine_direction(
            e_dict[s], params[s], val_r1[s], grads[s], grads_r1[s],
            dimer_axis[s], settings_list[s].radius, _single_eval,
        )
        val_r1[s] = vr1_new

    # --- Main loop ---
    max_cycles = max(s.max_iter for s in settings_list)
    for cycle in range(1, max_cycles + 1):
        active = [s for s in range(n_systems) if not converged[s]]
        if not active:
            break

        # Update f_para
        for s in active:
            f_para_old[s] = f_para[s].clone()
            f_para[s] = -grads[s].dot(dimer_axis[s]) * dimer_axis[s]

        # --- Rotation decision ---
        need_rotation = []
        need_r1_eval = []
        for s in active:
            cfg = settings_list[s]
            if cycle == 1 and not cfg.skip_first_rotation:
                need_rotation.append(s)
                need_r1_eval.append(s)
            elif cycle > 1 and not cfg.only_one_rotation:
                if cycle % interval_rot[s] == 0:
                    need_r1_eval.append(s)

        # Batched R1 eval for rotation candidates
        if need_r1_eval:
            r1_pos_sub = [(params[s] + settings_list[s].radius * dimer_axis[s]).reshape(-1, 3)
                          for s in need_r1_eval]
            e_r1_sub, g_r1_sub = _eval_subset(need_r1_eval, r1_pos_sub)
            for s in need_r1_eval:
                grads_r1[s] = g_r1_sub[s]
                val_r1[s] = e_r1_sub[s]
                curvature[s] = ((grads_r1[s] - grads[s]).dot(dimer_axis[s]) / settings_list[s].radius).item()

        # Check rotation condition for non-first-cycle systems
        for s in need_r1_eval:
            if s in need_rotation:
                continue
            ortho_fn_s = (
                2.0 * (grads_r1[s] - grads[s]).dot(dimer_axis[s]) * dimer_axis[s]
                - 2.0 * (grads_r1[s] - grads[s])
            )
            if ((curvature[s] > 0 and f_para[s].norm() < f_para_old[s].norm())
                    or (curvature[s] < 0 and f_para[s].norm() > f_para_old[s].norm())):
                need_rotation.append(s)
            elif ortho_fn_s.norm().item() > settings_list[s].rotation_force_threshold:
                need_rotation.append(s)

        # --- Perform rotation (batched R1 evals across rotating systems) ---
        first_cycle_rot = [s for s in need_rotation if cycle == 1]
        later_rot = [s for s in need_rotation if cycle > 1]

        for s in need_rotation:
            n_rot_performed[s] += 1

        # rotationWithGradient for later-cycle systems
        if later_rot:
            pre_rot_axis = {s: dimer_axis[s].clone() for s in later_rot}
            rot_thresh = {}
            m_counters = {}
            rot_converged_flag = {s: False for s in later_rot}
            old_r1 = {s: torch.zeros(n_params[s], device=device, dtype=dtype) for s in later_rot}

            for s in later_rot:
                cfg = settings_list[s]
                rt = cfg.rotation_gradient_threshold_other_cycles
                if cfg.decrease_rotation_gradient_threshold and n_rot_performed[s] > cfg.cycle_of_rotation_gradient_decrease:
                    rt = cfg.lowered_rotation_gradient_threshold
                rot_thresh[s] = rt
                m_counters[s] = 0

            ortho_g_norms = {}
            max_rc = max(settings_list[s].max_rotations_other_cycles for s in later_rot)
            for rc in range(max_rc):
                still_rotating = [s for s in later_rot if not rot_converged_flag[s]]
                if not still_rotating:
                    break

                for s in still_rotating:
                    ortho_fn_s = (
                        2.0 * (grads_r1[s] - grads[s]).dot(dimer_axis[s]) * dimer_axis[s]
                        - 2.0 * (grads_r1[s] - grads[s])
                    )
                    cfg = settings_list[s]
                    if cfg.rotation_cg:
                        ortho_g_s = _cg_rotation(ortho_fn_s, prev_ortho_fn[s], prev_ortho_g[s], rc)
                    elif cfg.rotation_lbfgs:
                        ortho_g_s, m_counters[s] = _lbfgs_rotation(
                            ortho_fn_s, prev_ortho_fn[s],
                            params[s] + cfg.radius * dimer_axis[s], old_r1[s],
                            dx_mat[s], dg_mat[s], rc, m_counters[s], cfg.lbfgs_memory,
                        )
                    else:
                        ortho_g_s = ortho_fn_s.clone()

                    prev_ortho_g[s] = ortho_g_s.clone()
                    prev_ortho_fn[s] = ortho_fn_s.clone()
                    old_r1[s] = (params[s] + cfg.radius * dimer_axis[s]).clone()
                    ortho_g_norms[s] = ortho_g_s.norm().item()

                    new_r1 = params[s] + cfg.radius * dimer_axis[s] + ortho_g_s
                    dimer_axis[s] = new_r1 - params[s]
                    dimer_axis[s] = dimer_axis[s] / dimer_axis[s].norm()

                # Batched R1 evaluation
                r1_pos_rot = [(params[s] + settings_list[s].radius * dimer_axis[s]).reshape(-1, 3)
                              for s in still_rotating]
                e_r1_rot, g_r1_rot = _eval_subset(still_rotating, r1_pos_rot)
                for s in still_rotating:
                    grads_r1[s] = g_r1_rot[s]
                    val_r1[s] = e_r1_rot[s]
                    curvature[s] = ((grads_r1[s] - grads[s]).dot(dimer_axis[s]) / settings_list[s].radius).item()

                # Convergence check uses ortho_g norm (step size), matching SCINE
                for s in still_rotating:
                    if ortho_g_norms[s] < rot_thresh[s]:
                        rot_converged_flag[s] = True
                        cos_sim = dimer_axis[s].dot(pre_rot_axis[s]).item()
                        if cos_sim > 0.99:
                            interval_rot[s] += 1

            # Direction determination for rotated systems
            for s in later_rot:
                def _single_eval(p, _s=s):
                    e_d, g_d = _eval_subset([_s], [p.reshape(-1, 3)])
                    return e_d[_s], g_d[_s]
                cur_val = e_dict.get(s, check_old_value[s])
                dimer_axis[s], grads_r1[s], _, vr1_new, curvature[s] = _determine_direction(
                    cur_val, params[s], val_r1[s], grads[s], grads_r1[s],
                    dimer_axis[s], settings_list[s].radius, _single_eval,
                )
                val_r1[s] = vr1_new
                if not rot_converged_flag[s]:
                    interval_rot[s] = 1

        # rotationWithPhi for first-cycle systems (same batched pattern)
        if first_cycle_rot:
            phi_rot_converged = {s: False for s in first_cycle_rot}
            phi_min_dict = {s: 0.0 for s in first_cycle_rot}
            phi1_dict = {s: 0.0 for s in first_cycle_rot}
            grads_r1_phi1 = {s: torch.zeros(n_params[s], device=device, dtype=dtype) for s in first_cycle_rot}
            m_counters = {s: 0 for s in first_cycle_rot}
            old_r1 = {s: torch.zeros(n_params[s], device=device, dtype=dtype) for s in first_cycle_rot}
            ortho_dimer = {s: dimer_axis[s].clone() for s in first_cycle_rot}

            max_rc = max(settings_list[s].max_rotations_first_cycle for s in first_cycle_rot)
            for rc in range(max_rc):
                still_rotating = [s for s in first_cycle_rot if not phi_rot_converged[s]]
                if not still_rotating:
                    break

                # R1 eval (batched, skip if interpolating)
                need_eval = []
                for s in still_rotating:
                    if settings_list[s].gradient_interpolation and rc != 0:
                        phi1 = phi1_dict[s]
                        pm = phi_min_dict[s]
                        grads_r1[s] = (
                            math.sin(phi1 - pm) / math.sin(phi1) * grads_r1[s]
                            + math.sin(pm) / math.sin(phi1) * grads_r1_phi1[s]
                            + (1 - math.cos(pm) - math.sin(pm) * math.tan(phi1 / 2)) * grads[s]
                        )
                    else:
                        need_eval.append(s)

                if need_eval:
                    r1_pos_ne = [(params[s] + settings_list[s].radius * dimer_axis[s]).reshape(-1, 3) for s in need_eval]
                    _, g_r1_ne = _eval_subset(need_eval, r1_pos_ne)
                    for s in need_eval:
                        grads_r1[s] = g_r1_ne[s]

                # Per-system rotation logic
                trial_needed = []
                trial_axes = {}
                for s in still_rotating:
                    cfg = settings_list[s]
                    ortho_dimer[s] = dimer_axis[s].clone()
                    ortho_fn_s = (
                        2.0 * (grads_r1[s] - grads[s]).dot(dimer_axis[s]) * dimer_axis[s]
                        - 2.0 * (grads_r1[s] - grads[s])
                    )
                    ortho_g_s = ortho_fn_s.clone()
                    if cfg.rotation_cg:
                        ortho_g_s = _cg_rotation(ortho_fn_s, prev_ortho_fn[s], prev_ortho_g[s], rc)
                    elif cfg.rotation_lbfgs:
                        ortho_g_s, m_counters[s] = _lbfgs_rotation(
                            ortho_fn_s, prev_ortho_fn[s],
                            params[s] + cfg.radius * dimer_axis[s], old_r1[s],
                            dx_mat[s], dg_mat[s], rc, m_counters[s], cfg.lbfgs_memory,
                        )
                    ortho_g_s = ortho_g_s - ortho_g_s.dot(dimer_axis[s]) * dimer_axis[s]
                    theta_norm = ortho_g_s.norm()
                    if theta_norm < 1e-20:
                        phi_rot_converged[s] = True
                        continue
                    theta = ortho_g_s / theta_norm
                    prev_ortho_g[s] = ortho_g_s.clone()
                    prev_ortho_fn[s] = ortho_fn_s.clone()
                    old_r1[s] = (params[s] + cfg.radius * dimer_axis[s]).clone()

                    c0 = ((grads_r1[s] - grads[s]).dot(dimer_axis[s]) / cfg.radius).item()
                    dc_dphi = (2.0 * (grads_r1[s] - grads[s]).dot(theta) / cfg.radius).item()

                    if abs(dc_dphi) < cfg.rotation_gradient_threshold_first_cycle:
                        phi_rot_converged[s] = True
                        if rc == 0:
                            interval_rot[s] += 1
                        continue

                    phi1 = -0.5 * math.atan2(dc_dphi, 2.0 * abs(c0))
                    phi1_dict[s] = phi1
                    if abs(phi1) < cfg.phi_tolerance:
                        phi_rot_converged[s] = True
                        continue

                    dimer_axis[s] = ortho_dimer[s] * math.cos(phi1) + theta * math.sin(phi1)
                    dimer_axis[s] = dimer_axis[s] / dimer_axis[s].norm()
                    trial_needed.append(s)
                    trial_axes[s] = theta

                # Batched trial eval
                if trial_needed:
                    r1_trial = [(params[s] + settings_list[s].radius * dimer_axis[s]).reshape(-1, 3) for s in trial_needed]
                    _, g_trial = _eval_subset(trial_needed, r1_trial)

                    for s in trial_needed:
                        cfg = settings_list[s]
                        grads_r1_phi1[s] = g_trial[s]
                        c0 = ((grads_r1[s] - grads[s]).dot(ortho_dimer[s]) / cfg.radius).item()
                        c_phi1 = ((grads_r1_phi1[s] - grads[s]).dot(dimer_axis[s]) / cfg.radius).item()
                        phi1 = phi1_dict[s]
                        dc_dphi = (2.0 * (grads_r1[s] - grads[s]).dot(trial_axes[s]) / cfg.radius).item()

                        b1 = 0.5 * dc_dphi
                        denom_fourier = 1 - math.cos(2 * phi1)
                        if abs(denom_fourier) < 1e-20:
                            denom_fourier = 1e-20
                        a1 = (c0 - c_phi1 + b1 * math.sin(2 * phi1)) / denom_fourier
                        a0 = 2 * (c0 - a1)
                        phi_min = 0.5 * math.atan2(b1, a1)
                        c_phi_min = a0 / 2.0 + a1 * math.cos(2.0 * phi_min) + b1 * math.sin(2.0 * phi_min)

                        if c_phi_min > c0 and c_phi_min > c_phi1:
                            phi_min += math.pi / 2
                            curvature[s] = a0 / 2.0 + a1 * math.cos(2.0 * phi_min) + b1 * math.sin(2.0 * phi_min)
                        elif c_phi1 < c_phi_min:
                            phi_min = phi1
                            curvature[s] = c_phi1
                        elif c0 < c_phi_min:
                            phi_min = 0.0
                            curvature[s] = c0

                        phi_min_dict[s] = phi_min
                        dimer_axis[s] = ortho_dimer[s]
                        if abs(phi_min) < cfg.phi_tolerance:
                            phi_rot_converged[s] = True
                        else:
                            theta = trial_axes[s]
                            dimer_axis[s] = ortho_dimer[s] * math.cos(phi_min) + theta * math.sin(phi_min)
                            dimer_axis[s] = dimer_axis[s] / dimer_axis[s].norm()

            # Direction determination for first-cycle rotated systems
            for s in first_cycle_rot:
                if not phi_rot_converged[s]:
                    r1_p = (params[s] + settings_list[s].radius * dimer_axis[s]).reshape(-1, 3)
                    e_fin, g_fin = _eval_subset([s], [r1_p])
                    grads_r1[s] = g_fin[s]
                    val_r1[s] = e_fin[s]
                def _single_eval(p, _s=s):
                    e_d, g_d = _eval_subset([_s], [p.reshape(-1, 3)])
                    return e_d[_s], g_d[_s]
                cur_val = e_dict.get(s, check_old_value[s])
                dimer_axis[s], grads_r1[s], _, vr1_new, curvature[s] = _determine_direction(
                    cur_val, params[s], val_r1[s], grads[s], grads_r1[s],
                    dimer_axis[s], settings_list[s].radius, _single_eval,
                )
                val_r1[s] = vr1_new

        for s in need_rotation:
            default_step[s] = settings_list[s].default_translation_step

        # --- Translation ---
        steps_dict = {}
        for s in active:
            cfg = settings_list[s]
            if curvature[s] > 0 and cycle < cfg.minimization_cycle:
                mod_g = -grads[s].dot(dimer_axis[s]) * dimer_axis[s]
            else:
                mod_g = -2.0 * (grads[s].dot(dimer_axis[s]) * dimer_axis[s]) + grads[s]

            if cycle == 1:
                old_params[s] = params[s].clone()
                old_grads[s] = mod_g.clone()
                sv = -mod_g
            else:
                dx = params[s] - old_params[s]
                dg = mod_g - old_grads[s]
                dx_dot_dg = dx.dot(dg).item()
                dg_inv_dg = (dg @ inv_h[s] @ dg).item()

                half_eye = 0.5 * torch.eye(n_params[s], device=device, dtype=dtype)
                if torch.allclose(inv_h[s], half_eye, atol=1e-10):
                    dg2 = dg.dot(dg).item()
                    if abs(dg2) > 1e-20:
                        inv_h[s].diagonal().mul_(2.0 * dx_dot_dg / dg2)

                sigma2, sigma3 = 0.9, 9.0
                dp = 1.0
                if abs(dx_dot_dg) < abs((1.0 - sigma2) * dg_inv_dg):
                    dp = sigma2 * dg_inv_dg / (dg_inv_dg - dx_dot_dg)
                elif abs(dx_dot_dg) > abs((1.0 + sigma3) * dg_inv_dg):
                    dp = -sigma3 * dg_inv_dg / (dg_inv_dg - dx_dot_dg)
                if abs(dp - 1.0) >= 1e-16:
                    dx = dp * dx + (1.0 - dp) * (inv_h[s] @ dg)
                    dx_dot_dg = dx.dot(dg).item()
                if abs(dx_dot_dg) < 1e-9:
                    dx_dot_dg = -1e-9 if dx_dot_dg < 0 else 1e-9

                ab = (dx_dot_dg + (dg @ inv_h[s] @ dg).item()) / (dx_dot_dg ** 2)
                bb = 1.0 / dx_dot_dg
                inv_h[s] = (
                    inv_h[s] + ab * torch.outer(dx, dx)
                    - bb * (inv_h[s] @ torch.outer(dg, dx) + torch.outer(dx, dg) @ inv_h[s])
                )
                old_params[s] = params[s].clone()
                old_grads[s] = mod_g.clone()

                if (not applied_stepsize_scaling[s]
                        and (cycle >= cfg.bfgs_start
                             or _gradient_below_threshold(grads[s], cfg.grad_rmsd_threshold))):
                    applied_stepsize_scaling[s] = True
                sv = -(inv_h[s] @ mod_g) if applied_stepsize_scaling[s] else -mod_g

            step = default_step[s] * sv
            mx = step.abs().max().item()
            if mx > cfg.trust_radius:
                step = step * (cfg.trust_radius / mx)
                inv_h[s] = 0.5 * torch.eye(n_params[s], device=device, dtype=dtype)
                applied_stepsize_scaling[s] = False
            params[s] = params[s] + step
            steps_dict[s] = step

        # --- Batched evaluation at new positions ---
        new_pos = [params[s].reshape(-1, 3) for s in active]
        e_new, g_new = _eval_subset(active, new_pos)
        e_dict = e_new
        for s in active:
            grads[s] = g_new[s]

        # --- Convergence + oscillation ---
        osc_systems = []
        for s in active:
            cfg = settings_list[s]
            delta_p = params[s] - check_old_params[s]
            delta_v = e_new[s] - check_old_value[s]
            check_old_params[s] = params[s].clone()
            check_old_value[s] = e_new[s]

            stop = cycle >= cfg.max_iter or gradient_based_converged(
                grads[s], delta_p, delta_v,
                cfg.grad_max_coeff, cfg.grad_rms,
                cfg.step_max_coeff, cfg.step_rms,
                cfg.delta_value, cfg.convergence_requirement,
            )
            if stop:
                converged[s] = True
                cycle_counts[s] = cycle
                continue

            value_memory[s].append(e_new[s])
            if _is_oscillating(value_memory[s]):
                params[s] = params[s] - steps_dict[s] / 2.0
                default_step[s] *= 0.95
                osc_systems.append(s)
            else:
                default_step[s] = 1.0

        # Batched re-eval for oscillating systems
        if osc_systems:
            osc_pos = [params[s].reshape(-1, 3) for s in osc_systems]
            e_osc, g_osc = _eval_subset(osc_systems, osc_pos)
            for s in osc_systems:
                grads[s] = g_osc[s]
                e_dict[s] = e_osc[s]
                check_old_value[s] = e_osc[s]
                check_old_params[s] = params[s].clone()

    # Mark unconverged systems
    for s in range(n_systems):
        if not converged[s]:
            cycle_counts[s] = settings_list[s].max_iter

    # --- Reassemble result ---
    result_positions = torch.cat([params[s].reshape(-1, 3) for s in range(n_systems)], dim=0)
    ts_state = SimState(
        positions=result_positions,
        masses=state.masses.clone(),
        cell=state.cell.clone(),
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers.clone(),
        system_idx=system_idx.clone(),
    )
    return ts_state, cycle_counts
