#!/usr/bin/env python3
"""Fig. 1: isolation-transmission frontier. Writes fig_frontier.pdf (and .png preview)."""
import numpy as np
from math import e, pi, sqrt
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#b4b2a9", "#2b2a27", "#6f6d66"
rng = np.random.default_rng(11)

def theta(j):
    if abs(j - 1) < 1e-12:
        return 1.0
    if j < 1:
        x = sqrt(1 - j * j); return np.arctanh(x) / x
    y = sqrt(j * j - 1); return np.arctan(y) / y

js = np.linspace(1, 6, 2000)
eps = np.array([abs(1 - j) * np.exp(-theta(j)) / j for j in js])
T = np.array([(1 + j) * np.exp(-theta(j)) / j for j in js])

def prop2(M, ts):
    """Exact 2x2 propagator via Cayley-Hamilton (stable at exceptional points)."""
    tr2 = (M[0, 0] + M[1, 1]) / 2
    Nm = M - tr2 * np.eye(2)
    w = np.sqrt(Nm[0, 0] ** 2 + Nm[0, 1] * Nm[1, 0] + 0j)
    sn = ts * np.sinc(w * ts / pi)
    return np.exp(-1j * tr2 * ts)[:, None, None] * (np.cos(w * ts)[:, None, None] * np.eye(2)
                                                     - 1j * sn[:, None, None] * Nm)

def sups(M, Tmax, nt=1200):
    U = prop2(M, np.linspace(0, Tmax, nt))
    return np.abs(U[:, 1, 0]).max(), np.abs(U[:, 0, 1]).max()

pts = []
for _ in range(6000):
    k1, k2 = 1.0, np.exp(rng.normal(0, 0.5))
    c = sqrt(k1 * k2) * rng.uniform() ** 0.3 * np.exp(2j * pi * rng.uniform())
    K = np.array([[k1, c], [np.conj(c), k2]])
    h12 = 0.5j * c + (rng.normal() + 1j * rng.normal()) * rng.uniform(0, 0.6) * abs(c)
    H = np.array([[0, h12], [np.conj(h12), rng.normal(0, 0.3)]])
    a21, a12 = sups(H - 0.5j * K, 25 / min(k1, k2))
    pts.append((a12, a21))
pts = np.array(pts)

aux = []
for kc in [10.0, 30.0, 100.0]:
    lam = sqrt(kc) / 2
    M3 = np.array([[0, 0.5j, lam], [-0.5j, 0, lam], [lam, lam, -0.5j * kc]])
    a21 = a12 = 0
    for t in np.linspace(0, 10, 3001):
        U = expm(-1j * M3 * t)
        a21 = max(a21, abs(U[1, 0])); a12 = max(a12, abs(U[0, 1]))
    aux.append((kc, a12, a21))

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
                     "axes.unicode_minus": False, "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(figsize=(3.375, 2.6))
ax.scatter(pts[:, 0], pts[:, 1], s=2, color=GRAY, lw=0, rasterized=True)
ax.scatter([], [], s=8, color=GRAY, lw=0, label="random two-mode generators")
ax.plot(eps, T, color=BLUE, lw=1.6, label=r"$\Phi(\epsilon)$, Prop. 2")
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)))
ax.scatter([0], [2 / e], s=28, color=BLUE, zorder=7, edgecolor="white", lw=0.8, clip_on=False)
ax.annotate(r"$2/e$ (Thm. 4)", (0, 2 / e), (0.03, 0.64), color=INK, fontsize=7,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"),
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
for kc, a12, a21 in aux:
    ax.scatter([a12], [a21], s=22, marker="D", color=ORANGE, zorder=6, edgecolor="white", lw=0.6)
ax.scatter([], [], s=22, marker="D", color=ORANGE, label=r"explicit auxiliary mode, $\kappa_c/\Gamma=10,30,100$")
ax.set_xlim(0, 0.55); ax.set_ylim(0.3, 1.08)
ax.set_xlabel(r"leakage $\epsilon=\sup_t|A_{1\leftarrow2}(t)|$")
ax.set_ylabel(r"transmission $\sup_t|A_{2\leftarrow1}(t)|$")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.legend(loc="upper left", fontsize=6.5, frameon=False, handletextpad=0.4)
fig.tight_layout(pad=0.3)
p0, p1 = ax.transData.transform((0.35, 0.35)), ax.transData.transform((0.5, 0.5))
ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
ax.text(0.40, 0.415, "reciprocal", color=INK, rotation=ang, fontsize=7, rotation_mode="anchor",
        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
fig.savefig("fig_frontier.pdf", dpi=300)
fig.savefig("fig_frontier.png", dpi=200)
print("aux points:", [(kc, round(a, 4), round(b, 4)) for kc, a, b in aux])
print("samples above frontier:", int(np.sum(pts[:, 1] > np.interp(pts[:, 0], eps, T) + 1e-6)))
