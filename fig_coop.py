#!/usr/bin/env python3
"""Fig. 3: scaled entanglement deficit sqrt(C)(1-E_N) versus cooperativity (exact, optimized). Writes fig_coop.pdf (and .png preview)."""
import numpy as np
from math import pi, sqrt, log, log2
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#2b2a27", "#6f6d66"

def EN(t, g, Dq, gq, gm):
    G = sqrt(2) * g
    M = np.array([[-0.5j * gm, G], [G, Dq - 0.5j * gq]])
    lam = np.linalg.eigvals(M); lp, lm = lam
    beta = ((M[0, 0] - lm) * np.exp(-1j * lp * t) - (M[0, 0] - lp) * np.exp(-1j * lm * t)) / (lp - lm)
    delta = np.exp(-0.5 * gm * t)
    a1, a2 = (beta + delta) / 2, (beta - delta) / 2
    p0 = 1 - abs(a1) ** 2 - abs(a2) ** 2
    return log2(1 + sqrt(p0 ** 2 + 4 * abs(a1 * a2) ** 2) - p0)

Cs = np.logspace(2, 6, 13)
opt, sub = [], []
for C in Cs:
    gm = 1 / C; Dp = sqrt(1 / gm); Ds = sqrt(2 / gm)
    r = minimize(lambda x: -EN(x[1] * pi * x[0] / 4, 1, x[0], 1, gm), [Dp, 1.0], method="Nelder-Mead",
                 options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 5000})
    opt.append(1 + r.fun)
    rs = minimize(lambda x: -EN(x[0] * pi * Ds / 4, 1, Ds, 1, gm), [1.0], method="Nelder-Mead",
                  options={"xatol": 1e-10, "fatol": 1e-14})
    sub.append(1 + rs.fun)
opt, sub = np.array(opt), np.array(sub)
c1, c2 = pi / (2 * log(2)), 3 * pi / (4 * sqrt(2) * log(2))
for C, o, s in zip(Cs, opt, sub):
    print(f"C={C:9.3g}  1-E_N opt={o:.5e} (sqrtC*={sqrt(C)*o:.4f})  fixed g*sqrt(2gq/gm)={s:.5e} (sqrtC*={sqrt(C)*s:.4f})")

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
                     "axes.unicode_minus": False, "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(figsize=(3.375, 2.3))
ax.axhline(c1, color=BLUE, lw=1.2)
ax.axhline(c2, color=ORANGE, lw=1.0, ls=(0, (4, 2)))
ax.semilogx(Cs, np.sqrt(Cs) * opt, "o-", ms=4, lw=0.8, color=BLUE, mec="white", mew=0.6)
ax.semilogx(Cs, np.sqrt(Cs) * sub, "s-", ms=3.5, lw=0.8, color=ORANGE, mec="white", mew=0.6)
ax.text(4e2, c1 - 0.03, r"$\pi/(2\ln 2)=2.266$: optimal detuning $g\sqrt{\gamma_q/\gamma_m}$",
        color=INK, fontsize=6.5, va="top")
ax.text(1.3e2, c2 + 0.012, r"$3\pi/(4\sqrt{2}\,\ln 2)=2.404$: detuning $g\sqrt{2\gamma_q/\gamma_m}$",
        color=INK, fontsize=6.5, va="bottom")
ax.set_xlabel(r"cooperativity $C=g^2/(\gamma_m\gamma_q)$")
ax.set_ylabel(r"$\sqrt{C}\,(1-E_N)$")
ax.set_ylim(2.15, 2.47)
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig("fig_coop.pdf"); fig.savefig("fig_coop.png", dpi=200)
