"""Simulate a rocket launched from Earth and save trajectory figures.

Two scenarios:
1. Rocket at near-escape velocity (v = 11.2 km/s) — barely escapes Earth,
   travels through the Earth-Moon system, interacts with lunar gravity.

2. Jacobi integral conservation — confirms RK4 accuracy.

Outputs (all under ``figures/``):
    rocket_trajectory.png   — 2-D inertial trajectory with Moon orbit
    rocket_jacobi.png       — Jacobi integral H'(t) vs time
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from rocket import simulate_rocket, rocket_ic, jacobi_integral, D_TL, OMEGA, R_T


FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)

N_STEPS    = 1_000_000
SAVE_EVERY = 500


def _polar_to_xy(rho, phi):
    return rho * np.cos(phi), rho * np.sin(phi)


def _moon_xy(t):
    return np.cos(OMEGA * t), np.sin(OMEGA * t)


def plot_trajectory(v_ms: float, theta: float, label: str, fname: str) -> None:
    rho0, phi0, p_rho0, p_phi0 = rocket_ic(v_ms=v_ms, theta=theta)
    print(f"  Simulating v={v_ms/1e3:.1f} km/s, theta={theta:.2f} rad ...")
    rho, phi, p_rho, p_phi, times = simulate_rocket(
        rho0, phi0, p_rho0, p_phi0, n_steps=N_STEPS, save_every=SAVE_EVERY
    )

    x, y = _polar_to_xy(rho, phi)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Moon orbit circle
    theta_moon = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta_moon), np.sin(theta_moon),
            "k--", lw=0.8, alpha=0.4, label="Moon orbit ($r = d_{TL}$)")

    # Rocket trajectory coloured by time
    points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap="plasma", linewidth=1.2,
                        norm=plt.Normalize(times[0], times[-1]))
    lc.set_array(times[:-1])
    ax.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label("time  [s]", fontsize=9)

    # Moon positions at several times
    for i, frac in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        t_mark = times[int(frac * (len(times) - 1))]
        mx, my = _moon_xy(t_mark)
        ax.plot(mx, my, "o", ms=6, color="silver",
                markeredgecolor="k", markeredgewidth=0.5,
                label="Moon" if i == 0 else None)

    # Earth (true radius too small to see; plot as small circle scaled 3x)
    earth_r = R_T / D_TL
    ax.add_patch(plt.Circle((0, 0), earth_r * 3, color="royalblue",
                             zorder=5, label="Earth (3× scaled)"))

    # Launch point
    x0, y0 = _polar_to_xy(rho0, phi0)
    ax.plot(x0, y0, "g^", ms=8, zorder=6, label="launch")

    ax.set_aspect("equal")
    lim = max(np.max(np.abs(x)), np.max(np.abs(y)), 1.5) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$x\;[d_{TL}]$", fontsize=11)
    ax.set_ylabel(r"$y\;[d_{TL}]$", fontsize=11)
    ax.set_title(
        f"Rocket trajectory — {label}\n"
        r"($d_{TL}$ = Earth-Moon distance, Earth at origin)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_jacobi(v_ms: float = 11.2e3, theta: float = 1.0) -> None:
    rho0, phi0, p_rho0, p_phi0 = rocket_ic(v_ms=v_ms, theta=theta)
    print("  Computing Jacobi integral ...")
    rho, phi, p_rho, p_phi, times = simulate_rocket(
        rho0, phi0, p_rho0, p_phi0, n_steps=N_STEPS, save_every=SAVE_EVERY
    )

    J = jacobi_integral(p_rho, p_phi, rho, phi, times)
    J0 = jacobi_integral(np.array([p_rho0]), np.array([p_phi0]),
                         np.array([rho0]), np.array([phi0]),
                         np.array([0.0]))[0]
    deviation = J - J0

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(times / 86400, deviation, lw=1.0, color="steelblue")
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("time  [days]", fontsize=10)
    ax.set_ylabel(r"$H' - H'_0$  $[\mathrm{m}^2\,\mathrm{s}^{-2}]$", fontsize=10)
    ax.set_title(
        f"Jacobi integral conservation — RK4  "
        f"($v_0 = {v_ms/1e3:.1f}$ km/s, $N_{{\\rm steps}} = {N_STEPS:,}$)",
        fontsize=11,
    )
    dev_max = np.max(np.abs(deviation))
    ax.text(0.98, 0.05, f"max $|\\Delta H'| = {dev_max:.2e}$",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "rocket_jacobi.png"), dpi=150)
    plt.close(fig)
    print("  saved rocket_jacobi.png")


def main():
    print("=== Rocket simulation ===")
    plot_trajectory(
        v_ms=11.2e3, theta=1.0,
        label=r"$v_0 = 11.2$ km/s (near-escape), $\theta = 1$ rad",
        fname="rocket_trajectory_escape.png",
    )
    plot_trajectory(
        v_ms=11.5e3, theta=1.0,
        label=r"$v_0 = 11.5$ km/s (super-escape), $\theta = 1$ rad",
        fname="rocket_trajectory_hyperbolic.png",
    )
    plot_jacobi()
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
