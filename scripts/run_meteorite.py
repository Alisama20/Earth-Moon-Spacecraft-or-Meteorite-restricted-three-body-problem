"""Simulate a meteorite approaching from deep space alongside an interceptor rocket.

Outputs (all under ``figures/``):
    meteorite_trajectory.png  — meteorite trajectory (wide-field view)
    meteorite_rocket.png      — both trajectories on the same plot
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rocket import simulate_both, rocket_ic, meteorite_ic, D_TL, OMEGA, R_T


FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)

N_STEPS    = 2_000_000
SAVE_EVERY = 2000


def _polar_to_xy(rho, phi):
    return rho * np.cos(phi), rho * np.sin(phi)


def _moon_xy(t):
    return np.cos(OMEGA * t), np.sin(OMEGA * t)


def run_simulation():
    rho0_m, phi0_m, p_rho0_m, p_phi0_m = meteorite_ic(
        rho0=10.0, phi0=np.pi/4, v_ms=5.0e3, theta=-3*np.pi/4
    )
    rho0_c, phi0_c, p_rho0_c, p_phi0_c = rocket_ic(
        v_ms=11.3e3, theta=np.pi/3, phi0=np.pi/2
    )

    print("  Simulating meteorite + rocket ...")
    (rho_m, phi_m, _, _,
     rho_c, phi_c, _, _, times) = simulate_both(
        rho0_m, phi0_m, p_rho0_m, p_phi0_m,
        rho0_c, phi0_c, p_rho0_c, p_phi0_c,
        n_steps=N_STEPS, save_every=SAVE_EVERY,
    )

    xm, ym = _polar_to_xy(rho_m, phi_m)
    xc, yc = _polar_to_xy(rho_c, phi_c)
    return xm, ym, xc, yc, times, rho0_m, phi0_m, rho0_c, phi0_c


def plot_meteorite(xm, ym, xc, yc, times, rho0_m, phi0_m, rho0_c, phi0_c) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    # Moon orbit
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8, alpha=0.4,
            label="Moon orbit")

    # Meteorite (wide trajectory)
    ax.plot(xm, ym, lw=1.0, color="tomato", alpha=0.8, label="Meteorite")
    # Mark start
    x0m, y0m = rho0_m * np.cos(phi0_m), rho0_m * np.sin(phi0_m)
    ax.plot(x0m, y0m, "rv", ms=8, zorder=5, label="Meteorite start")

    # Rocket trajectory
    ax.plot(xc, yc, lw=1.0, color="steelblue", alpha=0.8, label="Rocket")
    x0c, y0c = rho0_c * np.cos(phi0_c), rho0_c * np.sin(phi0_c)
    ax.plot(x0c, y0c, "b^", ms=8, zorder=5, label="Rocket launch")

    # Moon positions at start and end
    for frac in [0.0, 0.5, 1.0]:
        t_m = times[int(frac * (len(times) - 1))]
        mx, my = _moon_xy(t_m)
        ax.plot(mx, my, "o", ms=7, color="silver",
                markeredgecolor="k", markeredgewidth=0.6,
                label="Moon" if frac == 0.0 else None)

    # Earth
    earth_r = R_T / D_TL
    ax.add_patch(plt.Circle((0, 0), earth_r * 3, color="royalblue",
                             zorder=6, label="Earth (3× scaled)"))

    lim = 11.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x\;[d_{TL}]$", fontsize=11)
    ax.set_ylabel(r"$y\;[d_{TL}]$", fontsize=11)
    total_days = N_STEPS / 86400
    ax.set_title(
        f"Meteorite & rocket trajectories — {total_days:.0f}-day simulation\n"
        r"($v_m = 5$ km/s inward from $\rho=10$,  $v_c = 11.3$ km/s from Earth)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "meteorite_trajectory.png"), dpi=150)
    plt.close(fig)
    print("  saved meteorite_trajectory.png")


def plot_zoom(xm, ym, xc, yc, times, rho0_c, phi0_c) -> None:
    """Zoomed view of the Earth-Moon region."""
    fig, ax = plt.subplots(figsize=(7, 7))

    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8, alpha=0.4,
            label="Moon orbit")

    # Only plot the portion where both objects are within 3 d_TL
    mask_m = np.sqrt(xm**2 + ym**2) < 3.0
    mask_c = np.sqrt(xc**2 + yc**2) < 3.0

    ax.plot(xm[mask_m], ym[mask_m], lw=1.0, color="tomato",
            alpha=0.9, label="Meteorite")
    ax.plot(xc[mask_c], yc[mask_c], lw=1.0, color="steelblue",
            alpha=0.9, label="Rocket")

    for frac in [0.0, 0.33, 0.67, 1.0]:
        t_m = times[int(frac * (len(times) - 1))]
        mx, my = _moon_xy(t_m)
        ax.plot(mx, my, "o", ms=6, color="silver",
                markeredgecolor="k", markeredgewidth=0.6,
                label="Moon" if frac == 0.0 else None)

    earth_r = R_T / D_TL
    ax.add_patch(plt.Circle((0, 0), earth_r * 5, color="royalblue",
                             zorder=6, label="Earth (5× scaled)"))

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x\;[d_{TL}]$", fontsize=11)
    ax.set_ylabel(r"$y\;[d_{TL}]$", fontsize=11)
    ax.set_title(
        "Zoomed view — Earth-Moon neighbourhood\n"
        r"(trajectories clipped to $\rho < 3\,d_{TL}$)",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "meteorite_rocket_zoom.png"), dpi=150)
    plt.close(fig)
    print("  saved meteorite_rocket_zoom.png")


def main():
    print("=== Meteorite + rocket simulation ===")
    xm, ym, xc, yc, times, rho0_m, phi0_m, rho0_c, phi0_c = run_simulation()
    plot_meteorite(xm, ym, xc, yc, times, rho0_m, phi0_m, rho0_c, phi0_c)
    plot_zoom(xm, ym, xc, yc, times, rho0_c, phi0_c)
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
