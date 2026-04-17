"""Initial conditions and conserved quantities for the Earth-Moon 3-body problem."""

import numpy as np
from .dynamics import DELTA, MU, OMEGA, R_T, D_TL


def rocket_ic(v_ms: float = 11.2e3, theta: float = 1.0, phi0: float = np.pi / 2):
    """Launch conditions for a rocket from Earth's surface.

    Parameters
    ----------
    v_ms   : launch speed [m/s]
    theta  : velocity direction angle from the x-axis [rad] (inertial frame)
    phi0   : initial azimuthal position of the launch site [rad]

    Returns (rho0, phi0, p_rho0, p_phi0) in (d_TL, rad, d_TL/s, d_TL^2/s).
    """
    v = v_ms / D_TL
    rho0   = R_T / D_TL
    p_rho0 = v * np.cos(theta - phi0)
    p_phi0 = rho0 * v * np.sin(theta - phi0)
    return rho0, phi0, p_rho0, p_phi0


def meteorite_ic(rho0: float = 10.0, phi0: float = np.pi / 4,
                 v_ms: float = 5.0e3, theta: float = -3 * np.pi / 4):
    """Initial conditions for a meteorite arriving from deep space.

    Parameters
    ----------
    rho0  : initial distance from Earth [d_TL]
    phi0  : initial azimuthal angle [rad]
    v_ms  : speed [m/s]
    theta : velocity direction [rad]
    """
    v = v_ms / D_TL
    p_rho0 = v * np.cos(theta - phi0)
    p_phi0 = rho0 * v * np.sin(theta - phi0)
    return rho0, phi0, p_rho0, p_phi0


def jacobi_integral(p_rho, p_phi, rho, phi, t):
    """Jacobi integral J = H - omega * L_z (conserved in the co-rotating frame).

    Specific energy in the inertial frame (units: m^2/s^2):
        H = D_TL^2 * [p_rho^2/2 + p_phi^2/(2*rho^2) - DELTA/rho - mu*DELTA/rho_prime]

    Angular momentum per unit mass: L_z = p_phi * D_TL^2

    The Jacobi integral J = H - omega * L_z is exactly conserved when the Moon
    orbits at constant radius (circular orbit), because the co-rotating frame
    Hamiltonian is time-independent.
    """
    rho_prime = np.sqrt(1.0 + rho*rho - 2.0*rho*np.cos(phi - OMEGA*t))
    H = D_TL*D_TL * (p_rho*p_rho/2.0 + p_phi*p_phi/(2.0*rho*rho)
                     - DELTA/rho - MU*DELTA/rho_prime)
    return H - OMEGA * p_phi * D_TL * D_TL
