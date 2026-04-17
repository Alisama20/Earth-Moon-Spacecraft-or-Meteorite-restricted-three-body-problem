"""Equations of motion and RK4 integrator for the Earth-Moon restricted 3-body problem.

Coordinates: dimensionless polar (rho, phi), distances in units of d_TL.
Momenta: p_rho in d_TL/s, p_phi in d_TL^2/s.
Time in seconds.
"""

import numpy as np
from numba import njit

DELTA = 7.014744145e-12   # G*(M_Earth + M_Moon) / d_TL^3  [s^-2]
MU    = 0.01230246418     # M_Moon / (M_Earth + M_Moon)
OMEGA = 2.6617e-6         # Moon angular velocity [rad/s]
R_T   = 6.378160e6        # Earth radius [m]
D_TL  = 3.855e8           # Earth-Moon mean distance [m]


@njit(cache=True)
def _rhs(rho, phi, p_rho, p_phi, t):
    angle = phi - OMEGA * t
    rho_prime3 = (1.0 + rho*rho - 2.0*rho*np.cos(angle)) ** 1.5

    d_rho   = p_rho
    d_phi   = p_phi / (rho * rho)
    d_p_rho = (p_phi*p_phi / (rho**3)
               - DELTA * (1.0/(rho*rho) + MU*(rho - np.cos(angle))/rho_prime3))
    d_p_phi = -DELTA * MU * rho * np.sin(angle) / rho_prime3

    return d_rho, d_phi, d_p_rho, d_p_phi


@njit(cache=True)
def _rk4_step(rho, phi, p_rho, p_phi, t, h):
    k1r, k1f, k1pr, k1pf = _rhs(rho, phi, p_rho, p_phi, t)

    k2r, k2f, k2pr, k2pf = _rhs(
        rho + 0.5*h*k1r, phi + 0.5*h*k1f,
        p_rho + 0.5*h*k1pr, p_phi + 0.5*h*k1pf, t + 0.5*h)

    k3r, k3f, k3pr, k3pf = _rhs(
        rho + 0.5*h*k2r, phi + 0.5*h*k2f,
        p_rho + 0.5*h*k2pr, p_phi + 0.5*h*k2pf, t + 0.5*h)

    k4r, k4f, k4pr, k4pf = _rhs(
        rho + h*k3r, phi + h*k3f,
        p_rho + h*k3pr, p_phi + h*k3pf, t + h)

    h6 = h / 6.0
    return (rho   + h6*(k1r  + 2*k2r  + 2*k3r  + k4r),
            phi   + h6*(k1f  + 2*k2f  + 2*k3f  + k4f),
            p_rho + h6*(k1pr + 2*k2pr + 2*k3pr + k4pr),
            p_phi + h6*(k1pf + 2*k2pf + 2*k3pf + k4pf))


@njit(cache=True)
def _integrate_single(rho0, phi0, p_rho0, p_phi0, n_steps, save_every, h):
    n_out = n_steps // save_every
    rho_out   = np.empty(n_out)
    phi_out   = np.empty(n_out)
    p_rho_out = np.empty(n_out)
    p_phi_out = np.empty(n_out)
    t_out     = np.empty(n_out)

    rho, phi, p_rho, p_phi = rho0, phi0, p_rho0, p_phi0
    idx = 0
    for i in range(1, n_steps + 1):
        t = (i - 1) * h
        rho, phi, p_rho, p_phi = _rk4_step(rho, phi, p_rho, p_phi, t, h)
        if i % save_every == 0:
            rho_out[idx]   = rho
            phi_out[idx]   = phi
            p_rho_out[idx] = p_rho
            p_phi_out[idx] = p_phi
            t_out[idx]     = i * h
            idx += 1

    return rho_out, phi_out, p_rho_out, p_phi_out, t_out


@njit(cache=True)
def _integrate_two(
    rho0_m, phi0_m, p_rho0_m, p_phi0_m,
    rho0_c, phi0_c, p_rho0_c, p_phi0_c,
    n_steps, save_every, h
):
    n_out = n_steps // save_every
    rho_m_out   = np.empty(n_out); phi_m_out   = np.empty(n_out)
    p_rho_m_out = np.empty(n_out); p_phi_m_out = np.empty(n_out)
    rho_c_out   = np.empty(n_out); phi_c_out   = np.empty(n_out)
    p_rho_c_out = np.empty(n_out); p_phi_c_out = np.empty(n_out)
    t_out       = np.empty(n_out)

    rm, fm, prm, pfm = rho0_m, phi0_m, p_rho0_m, p_phi0_m
    rc, fc, prc, pfc = rho0_c, phi0_c, p_rho0_c, p_phi0_c
    idx = 0
    for i in range(1, n_steps + 1):
        t = (i - 1) * h
        rm, fm, prm, pfm = _rk4_step(rm, fm, prm, pfm, t, h)
        rc, fc, prc, pfc = _rk4_step(rc, fc, prc, pfc, t, h)
        if i % save_every == 0:
            rho_m_out[idx]   = rm;  phi_m_out[idx]   = fm
            p_rho_m_out[idx] = prm; p_phi_m_out[idx] = pfm
            rho_c_out[idx]   = rc;  phi_c_out[idx]   = fc
            p_rho_c_out[idx] = prc; p_phi_c_out[idx] = pfc
            t_out[idx]       = i * h
            idx += 1

    return (rho_m_out, phi_m_out, p_rho_m_out, p_phi_m_out,
            rho_c_out, phi_c_out, p_rho_c_out, p_phi_c_out, t_out)


def simulate_rocket(rho0, phi0, p_rho0, p_phi0,
                    n_steps: int = 1_000_000, save_every: int = 1000, h: float = 1.0):
    """Integrate a single spacecraft/particle in the Earth-Moon field."""
    return _integrate_single(rho0, phi0, p_rho0, p_phi0, n_steps, save_every, h)


def simulate_both(
    rho0_m, phi0_m, p_rho0_m, p_phi0_m,
    rho0_c, phi0_c, p_rho0_c, p_phi0_c,
    n_steps: int = 2_000_000, save_every: int = 2000, h: float = 1.0
):
    """Integrate meteorite and rocket simultaneously (independent particles)."""
    return _integrate_two(
        rho0_m, phi0_m, p_rho0_m, p_phi0_m,
        rho0_c, phi0_c, p_rho0_c, p_phi0_c,
        n_steps, save_every, h
    )
