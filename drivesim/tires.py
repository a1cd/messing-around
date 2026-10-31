"""Tire model: Pacejka-style Magic Formula shape with combined slip.

Pure-slip shape (normalised so that its peak equals 1):

    M(s) = sin(C * atan(B*s - E*(B*s - atan(B*s))))  /  max(M)

where ``s`` is a normalised slip magnitude.  Combined slip is handled with a
single slip vector (a "friction ellipse" style combination):

    kn = kappa / sigma_kappa          (normalised longitudinal slip ratio)
    an = alpha / sigma_alpha          (normalised lateral slip angle, rad)
    s  = sqrt(kn^2 + an^2)
    F  = D * M(s)                     (resultant tire force, D = mu * Fz)
    Fx = F * kn / s,   Fy = -F * an / s

so the resultant never exceeds the friction limit D and the longitudinal and
lateral forces trade off smoothly when both slips are present.

Load sensitivity: mu(Fz) = mu0 * (1 - lambda * (Fz/Fz0 - 1)), so that the
friction coefficient falls as load grows (as real tires do).

The shape constants (B=1.2, C=1.35, E=-3.0) were chosen so that the curve has
its peak at s = 1 (i.e. at sigma_kappa / sigma_alpha), a linear stiffness of
about 1.6 * D per unit normalised slip and a sliding level of about 0.87 of
peak (locked-wheel friction).

Everything here operates on Python floats; it is called once per wheel per
physics substep, so it avoids numpy on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def magic_raw(s: float, B: float, C: float, E: float) -> float:
    """Un-normalised Magic Formula shape for slip magnitude ``s >= 0``."""
    x = B * s
    return math.sin(C * math.atan(x - E * (x - math.atan(x))))


def _shape_peak(B: float, C: float, E: float) -> float:
    peak = 0.0
    for i in range(1, 4001):  # s in (0, 20]
        v = magic_raw(i * 0.005, B, C, E)
        if v > peak:
            peak = v
    return peak


@dataclass(frozen=True)
class TireParams:
    """Parameters of one tire / wheel assembly (SI units)."""

    radius: float = 0.33             # loaded rolling radius [m]
    wheel_inertia: float = 1.2       # rotational inertia of wheel+tire+brake disc [kg m^2]
    peak_mu: float = 1.05            # peak friction coefficient at nominal load on dry asphalt
    load_sensitivity: float = 0.12   # lambda in mu(Fz) = mu0*(1 - lambda*(Fz/Fz0 - 1))
    nominal_load: float = 3500.0     # Fz0 [N]
    sigma_kappa: float = 0.12        # longitudinal slip ratio at peak force
    sigma_alpha: float = 0.14        # lateral slip angle at peak force [rad]
    relax_length: float = 0.5        # lateral relaxation length [m]
    B: float = 1.2
    C: float = 1.35
    E: float = -3.0
    rolling_resistance: float = 0.012


class TireModel:
    """Stateless tire force evaluation (state such as relaxed slip lives in Vehicle)."""

    def __init__(self, params: TireParams | None = None):
        self.p = params or TireParams()
        p = self.p
        self.peak = _shape_peak(p.B, p.C, p.E)
        self.inv_peak = 1.0 / self.peak
        # Slope of the normalised shape at s = 0: d/ds[M] = B*C (normalised by peak).
        self.k0 = p.B * p.C * self.inv_peak

    def friction_limit(self, fz: float, surface: float) -> float:
        """Friction force limit D = mu(Fz) * Fz for a given normal load and surface factor."""
        if fz <= 0.0 or surface <= 0.0:
            return 0.0
        p = self.p
        ratio = fz / p.nominal_load
        mult = 1.0 - p.load_sensitivity * (ratio - 1.0)
        if mult < 0.6:
            mult = 0.6
        elif mult > 1.15:
            mult = 1.15
        return p.peak_mu * surface * mult * fz

    def combined_forces(self, D: float, kappa: float, alpha: float):
        """Return (Fx, Fy) in the tire frame for slip ratio ``kappa`` and slip angle ``alpha``.

        ``D`` is the friction limit (mu * Fz).  Fx is positive when the tire pushes the
        contact patch forward, Fy is positive to the left when alpha is positive
        (sign convention: the lateral force opposes the slip).
        """
        if D <= 0.0:
            return 0.0, 0.0
        p = self.p
        kn = kappa / p.sigma_kappa
        an = alpha / p.sigma_alpha
        s = math.sqrt(kn * kn + an * an)
        if s < 1e-9:
            return 0.0, 0.0
        f = D * magic_raw(s, p.B, p.C, p.E) * self.inv_peak / s
        return f * kn, -f * an
