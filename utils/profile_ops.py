from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve


@dataclass
class ProfileData:
    psi: np.ndarray
    value: np.ndarray
    label: str = ""
    path: Path | None = None


@dataclass
class TransformResult:
    psi: np.ndarray
    value: np.ndarray
    markers: list[float]
    summary: str
    patch_markers: list[float] = field(default_factory=list)


def read_prf(filename: str | Path, *, require_strict_psi: bool = True) -> ProfileData:
    path = Path(filename)
    with path.open("r") as f:
        first = f.readline().strip().split()
        if len(first) != 1:
            raise ValueError(f"{path}: first line should contain the point count")
        npts = int(first[0])

        psi = np.zeros(npts)
        value = np.zeros(npts)
        for i in range(npts):
            line = f.readline()
            if not line:
                raise EOFError(f"{path}: expected {npts} points, found {i}")
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}: line {i + 2} should contain psi and value")
            psi[i] = float(parts[0])
            value[i] = float(parts[1])

    _validate_profile(psi, value, require_strict_psi=require_strict_psi)
    return ProfileData(psi=psi, value=value, label=path.name, path=path)


def write_prf(filename: str | Path, psi: np.ndarray, value: np.ndarray) -> None:
    path = Path(filename)
    _validate_profile(psi, value)
    with path.open("w") as f:
        f.write(f"{len(psi)}\n")
        for p, v in zip(psi, value):
            f.write(f"{p:.12E}  {v:.12E}\n")
        f.write("-1\n")


def profile_derivatives(psi: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _validate_profile(psi, value)
    d1 = np.gradient(value, psi)
    d2 = np.gradient(d1, psi)
    return d1, d2


def scale_psi_axis(
    psi: np.ndarray,
    value: np.ndarray,
    *,
    delta_psi: float = -0.01,
    psi_ref: float = 1.0,
) -> TransformResult:
    _validate_profile(psi, value)
    if psi_ref == 0.0:
        raise ValueError("psi_ref must be nonzero")

    scale = (psi_ref - delta_psi) / psi_ref
    psi_new = scale * psi
    return TransformResult(
        psi=psi_new,
        value=value.copy(),
        markers=[psi_ref],
        summary=f"Shifted psi axis with delta_psi={delta_psi:g}, scale={scale:.8g}",
    )


def smooth_pedestal_top(
    psi: np.ndarray,
    value: np.ndarray,
    *,
    psi_start: float = 0.94,
    psi_end: float = 0.99,
    half_window: int = 4,
    poly_deg: int = 3,
    smooth_strength: float = 0.4,
    patch_width: float = 0.0,
    patch_passes: int = 0,
    patch_alpha: float = 0.25,
) -> TransformResult:
    _validate_profile(psi, value)
    if psi_start >= psi_end:
        raise ValueError("psi_start must be smaller than psi_end")
    if psi_start < psi.min() or psi_end > psi.max():
        raise ValueError("smoothing interval lies outside the profile grid")
    if half_window < 1:
        raise ValueError("half_window must be at least 1")
    if poly_deg < 1:
        raise ValueError("poly_deg must be at least 1")
    if not 0.0 <= smooth_strength <= 1.0:
        raise ValueError("smooth_strength must be between 0 and 1")
    if patch_width < 0.0:
        raise ValueError("patch_width must be non-negative")
    if patch_width > 0.0 and 2.0 * patch_width >= psi_end - psi_start:
        raise ValueError("patch_width must be smaller than half the smoothing interval")
    if patch_width > 0.0 and (psi_start - patch_width < psi.min() or psi_end + patch_width > psi.max()):
        raise ValueError("patch_width extends outside the profile grid")
    if patch_passes < 0:
        raise ValueError("patch_passes must be non-negative")
    if not 0.0 <= patch_alpha <= 0.5:
        raise ValueError("patch_alpha must be between 0 and 0.5")

    y0, dy0 = _local_value_and_slope(psi, value, psi_start, half_window, poly_deg)
    y1, dy1 = _local_value_and_slope(psi, value, psi_end, half_window, poly_deg)
    func, dy0_mod, dy1_mod, secant = _cubic_c1_segment(
        psi_start,
        psi_end,
        y0,
        dy0,
        y1,
        dy1,
        smooth_strength=smooth_strength,
    )
    summary = (
        "Smoothed pedestal top with C1 cubic: "
        f"dy_start={dy0:.3e}->{dy0_mod:.3e}, dy_end={dy1:.3e}->{dy1_mod:.3e}, "
        f"secant={secant:.3e}, smooth_strength={smooth_strength:g}"
    )
    value_new = value.copy()
    mask = (psi >= psi_start) & (psi <= psi_end)
    value_new[mask] = func(psi[mask])

    if patch_width > 0.0 and patch_passes > 0:
        _diffuse_patch_slice(value_new, psi, psi_start - patch_width, psi_start + patch_width, patch_passes, patch_alpha)
        _diffuse_patch_slice(value_new, psi, psi_end - patch_width, psi_end + patch_width, patch_passes, patch_alpha)
        summary += f", patch_width={patch_width:g}, patch_passes={patch_passes}, patch_alpha={patch_alpha:g}"
    patch_markers = (
        [psi_start - patch_width, psi_start + patch_width, psi_end - patch_width, psi_end + patch_width]
        if patch_width > 0.0 and patch_passes > 0
        else []
    )

    return TransformResult(
        psi=psi.copy(),
        value=value_new,
        markers=[psi_start, psi_end],
        summary=summary,
        patch_markers=patch_markers,
    )


def fixed_sep_tanh_connection(
    psi: np.ndarray,
    value: np.ndarray,
    *,
    psi_start: float = 0.95,
    psi_sep: float = 1.0,
    target_val_sep: float = 4.0e18,
    floor_val: float = 2.5e18,
    w_tanh: float = 0.05,
    psi_max: float = 1.3,
) -> TransformResult:
    _validate_fixed_sep_inputs(psi_start, psi_sep, floor_val, psi_max)
    if target_val_sep <= floor_val:
        raise ValueError("target_val_sep should be larger than floor_val")
    if w_tanh <= 0.0:
        raise ValueError("w_tanh must be positive")

    grad_start, val_start = _start_conditions(psi, value, psi_start)

    def equations(x_sym: float) -> float:
        z_st = (x_sym - psi_start) / w_tanh
        z_sep = (x_sym - psi_sep) / w_tanh
        sech2_st = 1.0 - np.tanh(z_st) ** 2
        if abs(sech2_st) < 1.0e-10:
            return 1.0e10
        amp = -grad_start * w_tanh / sech2_st
        return amp * (np.tanh(z_st) - np.tanh(z_sep)) - (val_start - target_val_sep)

    x_guess = (psi_start + psi_sep) / 2.0
    x_sol, _, ier, _ = fsolve(equations, x_guess, full_output=True)
    x_sym = float(x_sol[0]) if ier == 1 else x_guess

    z_st = (x_sym - psi_start) / w_tanh
    sech2_st = 1.0 - np.tanh(z_st) ** 2
    amp = -grad_start * w_tanh / sech2_st
    offset = val_start - amp * np.tanh(z_st)

    z_sep = (x_sym - psi_sep) / w_tanh
    sech2_sep = 1.0 - np.tanh(z_sep) ** 2
    grad_at_sep = -(amp / w_tanh) * sech2_sep
    amp_sol = target_val_sep - floor_val
    lam = 0.05 if (abs(grad_at_sep) < 1.0e-12 or amp_sol <= 0.0) else -amp_sol / grad_at_sep

    func_core_edge = lambda p: amp * np.tanh((x_sym - p) / w_tanh) + offset
    func_sol = lambda p: amp_sol * np.exp(-(p - psi_sep) / lam) + floor_val
    psi_new, value_new = _generate_fixed_sep_grid(
        psi,
        value,
        psi_start=psi_start,
        psi_sep=psi_sep,
        psi_max=psi_max,
        func_r1=func_core_edge,
        func_r2=func_sol,
    )

    return TransformResult(
        psi=psi_new,
        value=value_new,
        markers=[psi_start, psi_sep],
        summary=f"Fixed separatrix tanh: x_sym={x_sym:.5g}, lambda_SOL={lam:.5g}",
    )


def fixed_sep_exp_connection(
    psi: np.ndarray,
    value: np.ndarray,
    *,
    psi_start: float = 0.95,
    psi_sep: float = 1.0,
    target_val_sep: float | None = None,
    floor_val: float = 2.5e18,
    k_shape: float = -5.0,
    psi_max: float = 1.3,
) -> TransformResult:
    _validate_fixed_sep_inputs(psi_start, psi_sep, floor_val, psi_max)

    grad_start, val_start = _start_conditions(psi, value, psi_start)
    dx = psi_sep - psi_start

    if target_val_sep is not None:
        dval = target_val_sep - val_start

        def eq_k(k: float) -> float:
            if abs(grad_start) < 1.0e-9:
                return 1.0e9
            return np.exp(k * dx) - 1.0 - (k * dval / grad_start)

        k_sol, _, ier, _ = fsolve(eq_k, -5.0, full_output=True)
        k = float(k_sol[0]) if ier == 1 else -10.0
    else:
        k = k_shape

    if abs(k) < 1.0e-9:
        k = 1.0e-9

    amp = grad_start / k
    offset = val_start - amp
    if target_val_sep is None:
        target_val_sep = amp * np.exp(k * dx) + offset

    grad_at_sep = amp * k * np.exp(k * dx)
    amp_sol = target_val_sep - floor_val
    lam = 0.05 if (abs(grad_at_sep) < 1.0e-12 or amp_sol <= 0.0) else -amp_sol / grad_at_sep

    func_core_edge = lambda p: amp * np.exp(k * (p - psi_start)) + offset
    func_sol = lambda p: amp_sol * np.exp(-(p - psi_sep) / lam) + floor_val
    psi_new, value_new = _generate_fixed_sep_grid(
        psi,
        value,
        psi_start=psi_start,
        psi_sep=psi_sep,
        psi_max=psi_max,
        func_r1=func_core_edge,
        func_r2=func_sol,
    )

    return TransformResult(
        psi=psi_new,
        value=value_new,
        markers=[psi_start, psi_sep],
        summary=f"Fixed separatrix exponential: k={k:.5g}, lambda_SOL={lam:.5g}",
    )


def _validate_profile(psi: np.ndarray, value: np.ndarray, *, require_strict_psi: bool = True) -> None:
    if len(psi) != len(value):
        raise ValueError("psi and value must have the same length")
    if len(psi) < 3:
        raise ValueError("profile must contain at least 3 points")
    if not np.all(np.isfinite(psi)) or not np.all(np.isfinite(value)):
        raise ValueError("profile contains non-finite values")
    if require_strict_psi and np.any(np.diff(psi) <= 0.0):
        raise ValueError("psi grid must be strictly increasing")


def _validate_fixed_sep_inputs(
    psi_start: float,
    psi_sep: float,
    floor_val: float,
    psi_max: float,
) -> None:
    if psi_start >= psi_sep:
        raise ValueError("psi_start must be smaller than psi_sep")
    if psi_max <= psi_sep:
        raise ValueError("psi_max must be larger than psi_sep")
    if floor_val < 0.0:
        raise ValueError("floor_val should be non-negative")


def _local_value_and_slope(
    psi: np.ndarray,
    value: np.ndarray,
    psi_ref: float,
    half_window: int,
    poly_deg: int,
) -> tuple[float, float]:
    idx = int(np.argmin(np.abs(psi - psi_ref)))
    i0 = max(0, idx - half_window)
    i1 = min(len(psi), idx + half_window + 1)

    deg = min(poly_deg, i1 - i0 - 1)
    if deg < 1:
        raise ValueError("Need at least two local points to estimate C1 endpoint data")

    coef = np.polyfit(psi[i0:i1], value[i0:i1], deg)
    poly = np.poly1d(coef)
    return float(poly(psi_ref)), float(np.polyder(poly, 1)(psi_ref))


def _cubic_c1_segment(
    psi_start: float,
    psi_end: float,
    y0: float,
    dy0: float,
    y1: float,
    dy1: float,
    *,
    smooth_strength: float,
):
    dpsi = psi_end - psi_start
    secant = (y1 - y0) / dpsi
    dy0_mod = (1.0 - smooth_strength) * dy0 + smooth_strength * secant
    dy1_mod = (1.0 - smooth_strength) * dy1 + smooth_strength * secant

    def func(psi_eval: np.ndarray) -> np.ndarray:
        t = (psi_eval - psi_start) / dpsi
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        return h00 * y0 + h10 * dpsi * dy0_mod + h01 * y1 + h11 * dpsi * dy1_mod

    return func, dy0_mod, dy1_mod, secant


def _diffuse_patch_slice(
    value: np.ndarray,
    psi: np.ndarray,
    psi_left: float,
    psi_right: float,
    passes: int,
    alpha: float,
) -> None:
    indices = np.flatnonzero((psi >= psi_left) & (psi <= psi_right))
    if passes <= 0 or len(indices) < 3:
        return

    start = int(indices[0])
    stop = int(indices[-1]) + 1
    patch = value[start:stop].copy()
    tmp = patch.copy()
    for _ in range(passes):
        tmp[0] = patch[0]
        tmp[-1] = patch[-1]
        tmp[1:-1] = alpha * patch[:-2] + (1.0 - 2.0 * alpha) * patch[1:-1] + alpha * patch[2:]
        patch, tmp = tmp, patch
    value[start:stop] = patch


def _start_conditions(psi: np.ndarray, value: np.ndarray, psi_start: float) -> tuple[float, float]:
    _validate_profile(psi, value)
    if psi_start <= psi.min() or psi_start >= psi.max():
        raise ValueError("psi_start must lie inside the profile grid")

    idx = int(np.argmin(np.abs(psi - psi_start)))
    window = 3
    i0 = max(0, idx - window)
    i1 = min(len(psi), idx + window + 1)
    if i1 - i0 < 2:
        raise ValueError("not enough local points to estimate start conditions")

    coef = np.polyfit(psi[i0:i1], value[i0:i1], 1)
    return float(coef[0]), float(np.polyval(coef, psi_start))


def _generate_fixed_sep_grid(
    psi: np.ndarray,
    value: np.ndarray,
    *,
    psi_start: float,
    psi_sep: float,
    psi_max: float,
    func_r1,
    func_r2,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_profile(psi, value)

    mask_core = psi < psi_start
    psi_core = psi[mask_core]
    value_core = value[mask_core]
    psi_tail = psi[~mask_core]

    last_psi = float(psi_tail[-1]) if len(psi_tail) else psi_start
    if last_psi < psi_max:
        dpsi_tail = float(np.mean(np.diff(psi)[-5:])) if len(psi) > 5 else 0.001
        psi_extension = np.arange(last_psi + dpsi_tail, psi_max, dpsi_tail)
        psi_mod = np.concatenate([psi_tail, psi_extension])
    else:
        psi_mod = psi_tail

    value_mod = np.zeros_like(psi_mod)
    mask_r1 = psi_mod <= psi_sep
    value_mod[mask_r1] = func_r1(psi_mod[mask_r1])
    value_mod[~mask_r1] = func_r2(psi_mod[~mask_r1])

    return np.concatenate([psi_core, psi_mod]), np.concatenate([value_core, value_mod])
