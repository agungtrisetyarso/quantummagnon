#!/usr/bin/env python3
"""Reproduces every number in
"The price of isolation: sharp limits on non-reciprocal state transfer and
mediated entanglement in hybrid magnonic networks" (A. Trisetyarso).

Requirements: numpy, scipy (tested with numpy>=1.24, scipy>=1.10). Runtime: ~1 min.
Units: hbar = 1, g = 1 unless stated; rotating frame at the magnon frequency.
Sections:
  S1  Dark sector and dispersive sign            (Lemma 2, Remark 1, Lemma 3)
  S2  Dark-mode factorization (full Lindblad)    (Theorem 1)
  S3  Exact mediated entanglement               (Theorem 2, Corollary 1, Table I)
  S4  Protocol reversal and gauge criterion      (Lemma 4, Theorem 3(a), Proposition 1(i))
  S5  Isolation bound, two modes                 (Theorem 4(b),(c), Table II)
  S6  Leakage-transmission frontier             (Proposition 2, Conjecture 1, Fig. 1)
  S7  Theorem 4 from the full Lindblad equation  (Theorem 4(a),(d), Remark 3, Table III)
  S8  Channel-level reciprocity and its limits   (Theorem 3(b),(c), Proposition 1(ii)-(iv))
Fig. 1 is produced by fig_frontier.py; the analytic steps are checked by verify_symbolic.py.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from math import e, pi, sqrt, log2, log

rng = np.random.default_rng(20261017)
np.set_printoptions(precision=6, suppress=True)
TOL = 1e-10

def header(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)

# ---------------------------------------------------------------- helpers
def destroy(n):
    return np.diag(np.sqrt(np.arange(1, n)), 1)

def ops(N):
    """Two magnon modes (Fock cutoff N each) and a two-level mediator."""
    a, I, I2 = destroy(N), np.eye(N), np.eye(2)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)  # basis (g, e): sigma^- |e> = |g>
    m1 = np.kron(np.kron(a, I), I2)
    m2 = np.kron(np.kron(I, a), I2)
    s = np.kron(np.kron(I, I), sm)
    return m1, m2, s

def liouvillian(H, Ls):
    """Column-stacking vectorisation: vec(A X B) = (B^T kron A) vec(X)."""
    d = H.shape[0]
    I = np.eye(d)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for C in Ls:
        CdC = C.conj().T @ C
        L += np.kron(C.conj(), C) - 0.5 * np.kron(I, CdC) - 0.5 * np.kron(CdC.T, I)
    return L

def reduced_magnons(rho, N):
    r = rho.reshape(N, N, 2, N, N, 2)
    return np.einsum("abkcdk->abcd", r).reshape(N * N, N * N)

def partial_transpose_first(r, N):
    return r.reshape(N, N, N, N).transpose(2, 1, 0, 3).reshape(N * N, N * N)

def log_neg(r, N):
    return log2(np.abs(np.linalg.eigvalsh(partial_transpose_first(r, N))).sum())

def check_physical(r, label):
    herm = np.abs(r - r.conj().T).max()
    tr = abs(np.trace(r) - 1)
    mineig = np.linalg.eigvalsh((r + r.conj().T) / 2).min()
    ok = herm < 1e-9 and tr < 1e-9 and mineig > -1e-9
    print(f"   physicality[{label}]: |rho-rho^+|={herm:.1e}, |Tr-1|={tr:.1e}, "
          f"min eig={mineig:.1e} -> {'OK' if ok else 'FAIL'}")
    assert ok
    return ok

# ======================================================================
header("S1  Dark sector (Remark 1) and dispersive sign (Lemma 3)")
N = 4
m1, m2, s = ops(N)
g1, g2 = 1.0, 1.7
G = np.hypot(g1, g2)
b = (g1 * m1 + g2 * m2) / G

def ket(n1, n2):
    v = np.zeros(N * N * 2, complex)
    v[(n1 * N + n2) * 2 + 0] = 1  # mediator in |g>
    return v

for k in range(3):
    D = (g2 * ket(k, 0) - g1 * ket(0, k)) / G
    pred = sqrt(2 * k) * g1 * g2 / G**2 if k >= 2 else 0.0
    print(f"   k={k}: ||D_k||={np.linalg.norm(D):.4f}  ||b D_k||={np.linalg.norm(b @ D):.4f}"
          f"  (Remark 1 prediction {pred:.4f})")
dvec = (g2 * m1 - g1 * m2).conj().T / G @ ket(0, 0)
print(f"   corrected |D_1>: ||b D_1|| = {np.linalg.norm(b @ dvec):.1e}")

gg, Dq = 1.0, 50.0
Hs = np.array([[0, 0, gg], [0, 0, gg], [gg, gg, Dq]])
ev = np.linalg.eigvalsh(Hs)
print(f"   exact single-excitation levels (g=1, Delta_q=50): {ev}")
print(f"   Lemma 3 bright shift -G^2/Delta_q = {-2 * gg**2 / Dq:.4f}; dark shift 0")
assert abs(ev[0] - (-2 * gg**2 / Dq)) < 2e-3 and abs(ev[1]) < 1e-12

# ======================================================================
header("S2  Dark-mode factorization: full Lindblad, cutoff N=3 (Theorem 1)")
N = 3
m1, m2, s = ops(N)
gam, t = 0.01, 50.0
for gq in [1.0, 10.0, 100.0]:
    H = 10.0 * s.conj().T @ s + (m1.conj().T @ s + s.conj().T @ m1) + (m2.conj().T @ s + s.conj().T @ m2)
    L = liouvillian(H, [sqrt(gam) * m1, sqrt(gam) * m2, sqrt(gq) * s])
    psi = (m1.conj().T - m2.conj().T) @ np.eye(N * N * 2)[:, 0] / sqrt(2)
    rho0 = np.outer(psi, psi.conj())
    rho = (expm(L * t) @ rho0.reshape(-1, order="F")).reshape(N * N * 2, -1, order="F")
    check_physical(rho, f"gamma_q={gq}")
    pD = np.real(psi.conj() @ rho @ psi)
    print(f"   gamma_q={gq:6.1f}: P(D_1,t=50)={pD:.12f}   exp(-gamma_m t)={np.exp(-gam * t):.12f}")
    assert abs(pD - np.exp(-gam * t)) < 1e-9

# ======================================================================
header("S3  Mediated entanglement: closed form vs full Lindblad (Theorem 2, Corollary 1)")

def closed_form(t, g1, g2, Dq, gq, gm):
    """Theorem 2 amplitudes (rotating frame at omega_m), initial |1,0,g>."""
    G = np.hypot(g1, g2)
    M = np.array([[-0.5j * gm, G], [G, Dq - 0.5j * gq]])
    lam, V = np.linalg.eig(M)
    lp, lm = lam
    beta = ((M[0, 0] - lm) * np.exp(-1j * lp * t) - (M[0, 0] - lp) * np.exp(-1j * lm * t)) / (lp - lm)
    q = G * (np.exp(-1j * lp * t) - np.exp(-1j * lm * t)) / (lp - lm)
    delta = np.exp(-0.5 * gm * t)
    a1 = (g1**2 * beta + g2**2 * delta) / G**2
    a2 = g1 * g2 * (beta - delta) / G**2
    aq = g1 * q / G
    p0 = 1 - abs(a1)**2 - abs(a2)**2
    return a1, a2, aq, p0

def EN_formula(a1, a2, p0):
    return log2(1 + sqrt(p0**2 + 4 * abs(a1 * a2)**2) - p0)

def EN_closed(t, *p):
    a1, a2, aq, p0 = closed_form(t, *p)
    return EN_formula(a1, a2, p0)

def dispersive(g, Dq, gq, gm):
    """Corollary 1, at t* = pi |Delta_q| / (4 g^2)."""
    eta2 = np.exp(-pi * gq / (2 * abs(Dq)))
    xi2 = np.exp(-pi * gm * abs(Dq) / (4 * g**2))
    p0 = 1 - 0.5 * xi2 * (1 + eta2)
    a1a2 = 0.25 * xi2 * (1 + eta2)
    return log2(1 + sqrt(p0**2 + 4 * a1a2**2) - p0), p0, a1a2

def lindblad_EN(t, g1, g2, Dq, gq, gm, N):
    m1, m2, s = ops(N)
    H = Dq * s.conj().T @ s
    H = H + g1 * (m1.conj().T @ s + s.conj().T @ m1) + g2 * (m2.conj().T @ s + s.conj().T @ m2)
    L = liouvillian(H, [sqrt(gm) * m1, sqrt(gm) * m2, sqrt(gq) * s])
    psi = m1.conj().T @ np.eye(N * N * 2)[:, 0]
    rho0 = np.outer(psi, psi.conj())
    rho = (expm(L * t) @ rho0.reshape(-1, order="F")).reshape(N * N * 2, -1, order="F")
    r12 = reduced_magnons(rho, N)
    check_physical(rho, f"full, N={N}")
    check_physical(r12, f"rho_12, N={N}")
    return log_neg(r12, N), rho, r12

def t_opt(p, tmax):
    ts = np.linspace(0, tmax, 6001)
    vals = np.array([EN_closed(t, *p) for t in ts])
    i = int(vals.argmax())
    lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
    r = minimize_scalar(lambda t: -EN_closed(t, *p), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-10})
    return r.x, -r.fun

rows = [  # (gamma_q, gamma_m, Delta_q), g = 1
    (10.0, 0.0, 1000.0),
    (10.0, 1e-3, 140.0),
    (1.0, 1e-4, 100.0),    # optimal detuning g sqrt(gq/gm)  (Corollary 1)
]
table1 = []
for gq, gm, Dq in rows:
    p = (1.0, 1.0, Dq, gq, gm)
    ts = pi * Dq / 4
    a1, a2, aq, p0 = closed_form(ts, *p)
    norm_ok = (p0 >= abs(aq)**2 - 1e-12) and (0 <= p0 <= 1)
    disp, p0d, a1a2d = dispersive(1.0, Dq, gq, gm)
    Ec = EN_formula(a1, a2, p0)
    EL2, _, r12 = lindblad_EN(ts, *p, N=2)
    EL3, _, _ = lindblad_EN(ts, *p, N=3)
    # direct comparison of reduced density matrices
    rho_pred = np.zeros((4, 4), complex)
    v = np.zeros(4, complex); v[2] = a1; v[1] = a2           # |10> index 2, |01> index 1
    rho_pred += np.outer(v, v.conj()); rho_pred[0, 0] += p0
    drho = np.abs(rho_pred - r12).max()
    to, Eo = t_opt(p, 2 * ts)
    ELo, _, _ = lindblad_EN(to, *p, N=2)
    Cc = (1.0 / (gm * gq)) if gm > 0 else np.inf
    print(f"\n   row gamma_q={gq}, gamma_m={gm}, Delta_q={Dq}, C={Cc:.3g}")
    print(f"   t*={ts:.3f}: |a1|^2={abs(a1)**2:.6f} |a2|^2={abs(a2)**2:.6f} |aq|^2={abs(aq)**2:.2e}"
          f" p0={p0:.6f}  (p0>=|aq|^2: {norm_ok})")
    print(f"   max|rho12(closed form) - rho12(Lindblad)| = {drho:.2e}")
    print(f"   E_N(t*): Cor.1={disp:.4f}  closed form={Ec:.4f}  "
          f"Lindblad N=2={EL2:.4f}  N=3={EL3:.4f}")
    print(f"   optimal time t_opt={to:.3f}: closed form={Eo:.4f}  Lindblad={ELo:.4f}")
    assert drho < 1e-8 and abs(EL2 - EL3) < 1e-8 and abs(EL2 - Ec) < 1e-8 and norm_ok
    table1.append((gq, gm, Dq, disp, Ec, Eo, to))

# convergence of the dispersive formula toward the exact value
print("\n   dispersive error vs Delta_q (gamma_q=10, gamma_m=0):")
for Dq in [100, 300, 1000, 3000, 10000]:
    d, _, _ = dispersive(1.0, Dq, 10.0, 0.0)
    c = EN_closed(pi * Dq / 4, 1.0, 1.0, Dq, 10.0, 0.0)
    print(f"     Delta_q={Dq:6d}: |disp - exact| = {abs(d - c):.2e}   (g/Delta)^2 = {(1 / Dq)**2:.1e}")

# cooperativity scaling: optimal detuning Delta' = g sqrt(gq/gm) (Corollary 1), and full optimisation
print("\n   cooperativity scaling, gamma_q = 1:")
print("     C        sqrt(C)(1-E_N) at Delta'=g sqrt(gq/gm), t*   | optimised over (Delta,t): Delta_opt/Delta', t/t*, sqrt(C)(1-E_N)")
from scipy.optimize import minimize
for C in [1e2, 1e3, 1e4, 1e5, 1e6]:
    gm = 1.0 / C
    Dp = sqrt(1.0 / gm)
    c = EN_closed(pi * Dp / 4, 1.0, 1.0, Dp, 1.0, gm)
    f = lambda x: -EN_closed(x[1] * pi * x[0] / 4, 1.0, 1.0, x[0], 1.0, gm)
    r = minimize(f, [Dp, 1.0], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 5000})
    print(f"     {C:7.0e}   {sqrt(C) * (1 - c):.5f}                                  |"
          f"   {r.x[0] / Dp:.4f}   {r.x[1]:.4f}   {sqrt(C) * (1 + r.fun):.5f}")
    if C >= 1e4:
        assert abs(sqrt(C) * (1 + r.fun) - pi / (2 * log(2))) < 2e-3
Ds = sqrt(2 * 1.0 / 1e-6)
c_old = EN_closed(pi * Ds / 4, 1.0, 1.0, Ds, 1.0, 1e-6)
print(f"     reference: at Delta = g sqrt(2 gq/gm), C = 1e6: sqrt(C)(1-E_N) = {1e3 * (1 - c_old):.5f}"
      f"  [3pi/(4 sqrt2 ln2) = {3 * pi / (4 * sqrt(2) * log(2)):.5f}]")
print(f"     pi/(2 ln 2) = {pi / (2 * log(2)):.5f}")

# ======================================================================
header("S4  Protocol reversal (Theorem 3(a)) and gauge criterion (Lemma 4)")

def rand_sym(n, local_only=True):
    A = rng.normal(size=(n, n)); A = (A + A.T) / 2
    if local_only:
        K = np.diag(rng.uniform(0, 1, n))
    else:
        B = rng.normal(size=(n, n)); K = B @ B.T / n
    return A - 0.5j * K

def prop(Ms, taus):
    U = np.eye(Ms[0].shape[0], dtype=complex)
    for M, tau in zip(Ms, taus):
        U = expm(-1j * M * tau) @ U
    return U

worst_id, worst_pal = 0.0, 0.0
for trial in range(2000):
    n = 3
    k = rng.integers(2, 6)
    Ms = [rand_sym(n, local_only=bool(trial % 2)) for _ in range(k)]
    taus = rng.uniform(0.1, 2.0, k)
    U = prop(Ms, taus)
    Ur = prop(Ms[::-1], taus[::-1])
    worst_id = max(worst_id, abs(U[1, 0] - Ur[0, 1]))
    Mp, tp = Ms + Ms[-2::-1], np.concatenate([taus, taus[-2::-1]])
    Up = prop(Mp, tp)
    worst_pal = max(worst_pal, abs(abs(Up[1, 0]) - abs(Up[0, 1])))
print(f"   max |A21[P] - A12[P_rev]| over 2000 random symmetric protocols = {worst_id:.1e}")
print(f"   max ||A21|-|A12|| over 2000 palindromic protocols           = {worst_pal:.1e}")
assert worst_id < 1e-10 and worst_pal < 1e-10

# Example 1: symmetric generator, non-palindromic protocol, non-reciprocal
Mex = [np.array([[-0.05j, 0, 1.0], [0, -0.05j, 0.6], [1.0, 0.6, 2.0 - 0.5j]]),
       np.array([[-0.05j, 0, 0.3], [0, -0.05j, 1.2], [0.3, 1.2, -1.0 - 0.5j]])]
tex = np.array([1.1, 0.7])
U = prop(Mex, tex)
print(f"   Example 1 (symmetric, non-palindromic): |A21|^2={abs(U[1, 0])**2:.6f}  |A12|^2={abs(U[0, 1])**2:.6f}")
Ur = prop(Mex[::-1], tex[::-1])
print(f"                    reversed protocol:     |A12[P_rev]|^2={abs(Ur[0, 1])**2:.6f}  (= |A21[P]|^2)")

# Lemma 4: flux criterion on a triangle (2 magnons + mediator), time independent
print("   triangle with flux Phi, local loss only, t = 2.0:")
for Phi in [0, pi / 2, pi, 2 * pi / 3]:
    H = np.array([[0, 1.0 * np.exp(1j * Phi), 1.0], [np.exp(-1j * Phi), 0, 1.0], [1.0, 1.0, 0.5]], complex)
    M = H - 0.5j * np.diag([0.02, 0.02, 0.4])
    U = expm(-2.0j * M)
    print(f"     Phi={Phi:5.3f}: |A21|^2={abs(U[1, 0])**2:.6f} |A12|^2={abs(U[0, 1])**2:.6f}")
# two modes, coherent + correlated dissipation: mixed loop phase
print("   two modes, H12 = J e^{i phi}, K = Gamma [[1,1],[1,1]], J = Gamma/2 = 0.5:")
for phi in [0, pi / 4, pi / 2]:
    M = np.array([[0, 0.5 * np.exp(1j * phi)], [0.5 * np.exp(-1j * phi), 0]]) - 0.5j * np.ones((2, 2))
    U = expm(-2.0j * M)
    print(f"     phi={phi:5.3f}: |A21|^2={abs(U[1, 0])**2:.6f} |A12|^2={abs(U[0, 1])**2:.2e}")

# ======================================================================
header("S5  The 2/e bound for two-mode isolators (Theorem 4(b),(c), Table II)")

def isolator(k1, k2, h1, h2, c):
    """Most general isolating 2x2 generator: K PSD, H12 = i K12 / 2."""
    K = np.array([[k1, c], [np.conj(c), k2]])
    H = np.array([[h1, 0.5j * c], [np.conj(0.5j * c), h2]])
    return H - 0.5j * K, K

def sup_A21(M, kmin):
    f = lambda t: -abs(expm(-1j * M * t)[1, 0])
    ts = np.linspace(0, 12 / kmin, 1500)
    vals = [-f(t) for t in ts]
    i = int(np.argmax(vals))
    r = minimize_scalar(f, bounds=(ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]), method="bounded",
                        options={"xatol": 1e-12})
    return -r.fun, r.x

best = 0.0
viol = 0
for _ in range(5000):
    k1, k2 = rng.uniform(0.05, 3.0, 2)
    c = rng.uniform(0, 1) ** 0.25 * sqrt(k1 * k2) * np.exp(2j * pi * rng.uniform())
    h1, h2 = rng.normal(0, 0.5, 2)
    M, K = isolator(k1, k2, h1, h2, c)
    assert np.linalg.eigvalsh(K).min() > -1e-12 and abs(M[0, 1]) < 1e-15
    A, t = sup_A21(M, min(k1, k2))
    best = max(best, A)
    viol += A > 2 / e + 1e-12
    # cascaded form: M21 = -i K21 and |M21| <= sqrt(k1 k2)
    assert abs(M[1, 0] + 1j * K[1, 0]) < 1e-14
print(f"   5000 random isolating generators: max sup_t|A21| = {best:.6f}; 2/e = {2 / e:.6f}; violations = {viol}")

print("   Table II (sup_t |A21| for structured families; analytic value in brackets):")
for k1, k2 in [(1, 1), (1, 2), (1, 4), (1, 16)]:
    M, _ = isolator(k1, k2, 0, 0, sqrt(k1 * k2))
    A, t = sup_A21(M, min(k1, k2))
    print(f"     k1={k1}, k2={k2}: sup|A21|={A:.6f} at t={t:.4f}"
          f"   [Eq. (supexact): {2 * (k2 / k1) ** ((1 + k2 / k1) / (2 * (1 - k2 / k1))) if k1 != k2 else 2 / e:.6f}]")
for dh in [0, 0.5, 1.0]:
    M, _ = isolator(1, 1, 0, dh, 1.0)
    A, t = sup_A21(M, 1)
    print(f"     k1=k2=1, detuning h2-h1={dh}: sup|A21|={A:.6f}  (eq. requires 0)")
for r in [1.0, 0.8, 0.5]:
    M, _ = isolator(1, 1, 0, 0, r)
    A, t = sup_A21(M, 1)
    print(f"     k1=k2=1, |K12|={r}: sup|A21|={A:.6f}  analytic {2 * r / e:.6f}")

# Metelmann-Clerk construction with the auxiliary mode kept explicitly
print("   reservoir-engineered isolator with explicit auxiliary mode (Gamma = 4 lam^2/kappa_c = 1, gamma_m = 0):")
for kc in [10.0, 100.0, 1000.0]:
    lam = sqrt(kc) / 2
    J = 0.5
    M3 = np.array([[0, 1j * J, lam], [-1j * J, 0, lam], [lam, lam, -0.5j * kc]])
    best21, best12 = 0, 0
    for t in np.linspace(0, 10, 4001):
        U = expm(-1j * M3 * t)
        best21 = max(best21, abs(U[1, 0])); best12 = max(best12, abs(U[0, 1]))
    print(f"     kappa_c={kc:6.0f}: sup|A21|={best21:.6f}  sup|A12|={best12:.2e}   (2/e = {2 / e:.6f})")
Fmax = 0.5 + (2 / e) / 3 + (2 / e) ** 2 / 6
print(f"   F_bar bound = 1/2 + 2/(3e) + 2/(3e^2) = {Fmax:.6f}")

# ======================================================================
header("S6  Leakage-transmission frontier (Proposition 2, Conjecture 1)")

def theta(j):
    if abs(j - 1) < 1e-12:
        return 1.0
    if j < 1:
        x = sqrt(1 - j * j); return np.arctanh(x) / x
    y = sqrt(j * j - 1); return np.arctan(y) / y

def family_closed(j):
    f = np.exp(-theta(j)) / j
    return abs(1 - j) * f, (1 + j) * f        # (sup|A12|, sup|A21|)

def family_numeric(j, Gam=1.0):
    """Direct matrix exponential (robust at the exceptional point j = 1)."""
    J = j * Gam / 2
    M = np.array([[0, 1j * J], [-1j * J, 0]]) - 0.5j * Gam * np.ones((2, 2))
    out = []
    for (r, c) in [(0, 1), (1, 0)]:
        f = lambda t: -abs(expm(-1j * M * t)[r, c])
        ts = np.linspace(0, 20, 2001)
        v = [-f(t) for t in ts]
        i = int(np.argmax(v))
        res = minimize_scalar(f, bounds=(ts[max(i - 1, 0)], ts[min(i + 1, 2000)]), method="bounded",
                              options={"xatol": 1e-12})
        out.append(-res.fun)
    return out[0], out[1]

print("   closed form vs numerics for the reservoir-engineered family (j = 2J/Gamma):")
for j in [0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0]:
    c12, c21 = family_closed(j); n12, n21 = family_numeric(j)
    print(f"     j={j:4.2f}: closed (eps={c12:.6f}, T={c21:.6f})  numeric (eps={n12:.6f}, T={n21:.6f})")
    assert abs(c12 - n12) < 1e-6 and abs(c21 - n21) < 1e-6

js = np.linspace(1.0, 4.0, 3001)
curve = np.array([family_closed(j) for j in js])
def frontier(eps):
    return np.interp(eps, curve[:, 0], curve[:, 1])
print("   frontier Phi(eps) (branch j >= 1):")
for eps in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
    print(f"     eps={eps:5.2f}: Phi={frontier(eps):.4f}")

def prop2(M, ts):
    """Exact 2x2 propagator exp(-i M t) via Cayley-Hamilton; stable at exceptional points."""
    tr2 = (M[0, 0] + M[1, 1]) / 2
    Nm = M - tr2 * np.eye(2)
    w2 = Nm[0, 0] ** 2 + Nm[0, 1] * Nm[1, 0]
    w = np.sqrt(w2 + 0j)
    ph = np.exp(-1j * tr2 * ts)
    cs = np.cos(w * ts)
    sn = ts * np.sinc(w * ts / pi)                 # sin(w t)/w, finite as w -> 0
    U = ph[:, None, None] * (cs[:, None, None] * np.eye(2) - 1j * sn[:, None, None] * Nm)
    return U

def sups(M, T, nt=1500):
    ts = np.linspace(0, T, nt)
    U = prop2(M, ts)
    return np.abs(U[:, 1, 0]).max(), np.abs(U[:, 0, 1]).max()

def sups_refined(M, T):
    out = []
    for (r_, c_) in [(1, 0), (0, 1)]:
        ts = np.linspace(0, T, 3000)
        v = np.abs(prop2(M, ts)[:, r_, c_]); i = int(v.argmax())
        f = lambda t: -abs(prop2(M, np.array([t]))[0, r_, c_])
        res = minimize_scalar(f, bounds=(ts[max(i - 1, 0)], ts[min(i + 1, 2999)]), method="bounded",
                              options={"xatol": 1e-12})
        out.append(max(-res.fun, v.max()))
    return out[0], out[1]

# self-test of the closed-form propagator, including the exceptional point of the family at j = 1
for Mt in [np.array([[0, 0.5j], [-0.5j, 0]]) - 0.5j * np.ones((2, 2)),
           np.array([[0.3 - 0.2j, 0.4 + 0.1j], [0.2, -0.1 - 0.7j]])]:
    for tt in [0.3, 1.7, 4.0]:
        assert np.abs(prop2(Mt, np.array([tt]))[0] - expm(-1j * Mt * tt)).max() < 1e-12
print("   closed-form 2x2 propagator agrees with expm to 1e-12 (incl. exceptional point)")

def search(nsamp, mode):
    cand = []
    for it in range(nsamp):
        if mode == "broad":
            k1, k2 = 1.0, np.exp(rng.normal(0, 0.5))
            c = sqrt(k1 * k2) * rng.uniform() ** 0.3 * np.exp(2j * pi * rng.uniform())
            spread, hd = 0.5, 0.3
        else:      # targeted: near-rank-one K, small detuning, near the isolation point
            k1, k2 = 1.0, np.exp(rng.normal(0, 0.3))
            c = sqrt(k1 * k2) * rng.uniform(0.9, 1.0) * np.exp(2j * pi * rng.uniform())
            spread, hd = 0.3, 0.1
        K = np.array([[k1, c], [np.conj(c), k2]])
        h12 = 0.5j * c + (rng.normal() + 1j * rng.normal()) * rng.uniform(0, spread) * abs(c)
        H = np.array([[0, h12], [np.conj(h12), rng.normal(0, hd)]])
        M = H - 0.5j * K; T = 25 / min(k1, k2)
        a21, a12 = sups(M, T)
        if a12 < 0.3:
            cand.append((a21 - frontier(a12), M, T))
    cand.sort(key=lambda z: -z[0])
    worst = -1.0
    for _, M, T in cand[:50]:        # refine the 50 closest candidates with continuous maximisation
        a21, a12 = sups_refined(M, T)
        worst = max(worst, a21 - frontier(a12))
    return len(cand), max(worst, cand[0][0])

n, excess = search(40000, "broad")
print(f"   broad search: {n} random two-mode generators with sup|A12|<0.3: "
      f"max [sup|A21| - Phi(sup|A12|)] = {excess:.2e}")
n2, excess2 = search(30000, "targeted")
print(f"   targeted search: {n2} generators: max [sup|A21| - Phi(sup|A12|)] = {excess2:.2e}")
print(f"   explicit auxiliary mode, kappa_c = 10: Phi(0.0753) = {frontier(0.0753):.4f} (observed 0.7716)")

# ======================================================================
header("S7  Theorem 4 from the full Lindblad equation; dephasing; thermal baths; >2 modes")
from scipy.sparse.linalg import expm_multiply
import scipy.sparse as sps
from scipy.special import factorial

def sdestroy(n):
    return sps.diags(np.sqrt(np.arange(1, n)), 1).tocsr()

def sliouv(H, Ls):
    d = H.shape[0]; I_ = sps.eye(d, format="csr")
    L = -1j * (sps.kron(I_, H) - sps.kron(H.T, I_))
    for C in Ls:
        CdC = C.conj().T @ C
        L = L + sps.kron(C.conj(), C) - 0.5 * sps.kron(I_, CdC) - 0.5 * sps.kron(CdC.T, I_)
    return L.tocsc()

# (i) cascaded form of Theorem 4(a), full density matrix, single-photon input, Fock cutoff 3
Nf = 3
a_ = sdestroy(Nf); I_ = sps.eye(Nf)
m1s = sps.kron(a_, I_).tocsr(); m2s = sps.kron(I_, a_).tocsr()
phi_ = 0.7
Lc = m1s + np.exp(1j * phi_) * m2s
Hc = 0.5j * (np.exp(1j * phi_) * m1s.conj().T @ m2s - np.exp(-1j * phi_) * m2s.conj().T @ m1s)
Lv = sliouv(Hc, [Lc])
d = Nf * Nf
def fock2(n1, n2):
    v = np.zeros(d); v[n1 * Nf + n2] = 1; return v
for lab, psi, obs in [("1->2", fock2(1, 0), (m2s.conj().T @ m2s).toarray()),
                      ("2->1", fock2(0, 1), (m1s.conj().T @ m1s).toarray())]:
    rho0 = np.outer(psi, psi).reshape(-1, order="F")
    ts = np.linspace(0, 6, 601)
    vals = expm_multiply(Lv, rho0, start=0, stop=6, num=601, endpoint=True)
    best = max(np.real(np.trace(obs @ v.reshape(d, d, order="F"))) for v in vals)
    print(f"   cascaded master equation, {lab}: max receiver population = {best:.8f}")
print(f"   4/e^2 = {4 / e**2:.8f}")

# (ii) coherent input and thermal baths: mean-field transmissivity unchanged; Fock-cutoff convergence
for nth in [0.0, 0.2]:
    for Nf in [4, 6, 8, 10]:
        a_ = sdestroy(Nf); I_ = sps.eye(Nf)
        m1s = sps.kron(a_, I_).tocsr(); m2s = sps.kron(I_, a_).tocsr()
        Lc = m1s + m2s; Hc = 0.5j * (m1s.conj().T @ m2s - m2s.conj().T @ m1s)
        Ls = [sqrt(1 + nth) * Lc] + ([sqrt(nth) * Lc.conj().T] if nth > 0 else [])
        Lv = sliouv(Hc, Ls); al = 0.3
        coh = np.array([np.exp(-al**2 / 2) * al**n_ / np.sqrt(factorial(n_)) for n_ in range(Nf)])
        th = np.array([nth**n_ / (1 + nth)**(n_ + 1) for n_ in range(Nf)]) if nth > 0 else np.eye(Nf)[0]
        rho0 = np.kron(np.outer(coh, coh), np.diag(th))
        v = expm_multiply(Lv * 2.0, rho0.reshape(-1, order="F"))
        r = v.reshape(Nf * Nf, Nf * Nf, order="F")
        print(f"   n_th={nth}, Fock cutoff {Nf:2d}: |<m2>(t=2)|/alpha = {abs((m2s @ r).trace()) / al:.6f}")
print(f"   2/e = {2 / e:.6f}")

# (iii) local pure dephasing added to random perfect isolators: transfer probability stays <= 4/e^2
def dephased_transfer(M, gph, T, nt=600):
    I2 = np.eye(2)
    S = -1j * (np.kron(I2, M) - np.kron(M.conj(), I2))
    for jj, gg in enumerate(gph):
        P = np.zeros((2, 2)); P[jj, jj] = 1
        S = S + gg * (np.kron(P, P) - 0.5 * np.kron(I2, P) - 0.5 * np.kron(P, I2))
    step = expm(S * T / nt)
    best, leak = 0.0, 0.0
    for start, idx_out in [((0, 0), (1, 1)), ((1, 1), (0, 0))]:
        r = np.zeros((2, 2), complex); r[start] = 1; v = r.reshape(-1, order="F")
        for _ in range(nt):
            v = step @ v
            val = v.reshape(2, 2, order="F")[idx_out].real
            if start == (0, 0): best = max(best, val)
            else: leak = max(leak, abs(val))
    return best, leak
mxp, mxl = 0.0, 0.0
for _ in range(1500):
    k1, k2 = rng.uniform(0.1, 2, 2)
    c = sqrt(k1 * k2) * rng.uniform() ** 0.2 * np.exp(2j * pi * rng.uniform())
    K = np.array([[k1, c], [np.conj(c), k2]])
    Hm = np.array([[0, 0.5j * c], [np.conj(0.5j * c), rng.normal(0, 0.3)]])
    gph = rng.exponential(0.5, 2) * (rng.uniform(size=2) < 0.8)
    p_, l_ = dephased_transfer(Hm - 0.5j * K, gph, 15 / min(k1, k2))
    mxp = max(mxp, p_); mxl = max(mxl, l_)
print(f"   1500 isolators with random local dephasing: max transfer probability = {mxp:.5f}"
      f" (4/e^2 = {4 / e**2:.5f}); max reverse population = {mxl:.1e}")
assert mxp <= 4 / e**2 + 1e-9 and mxl < 1e-12

# (iv) more than two modes: static cascaded chains with internal modes (parameters from an optimisation,
#      evaluated here with matrix exponentials only)
def casc(rates, dets):
    c = np.sqrt(np.asarray(rates, float)).astype(complex); K = np.outer(c.conj(), c); n_ = len(rates)
    H = np.diag(dets).astype(complex)
    for i in range(n_):
        for j_ in range(i + 1, n_):
            H[i, j_] = 0.5j * K[i, j_]; H[j_, i] = np.conj(H[i, j_])
    return H - 0.5j * K
def sup_last_first(M, T=30, dt=0.005):
    n_ = M.shape[0]; S = expm(-1j * M * dt); U = np.eye(n_, dtype=complex); best = (0, 0, 0)
    for k_ in range(int(T / dt)):
        U = S @ U
        if abs(U[n_ - 1, 0]) > best[0]: best = (abs(U[n_ - 1, 0]), (k_ + 1) * dt, abs(U[0, n_ - 1]))
    return best
chains = {2: ([1, 1], [0, 0]),
          3: ([1, 1, 1], [0, 2.200006, 0]),
          4: ([1, 0.862857, 0.862857, 1], [0, 1.836017, -1.836017, 0]),
          5: ([1, 0.822902, 1, 0.822902, 1], [0, -3.463223, -1.741706, 1.721517, 0])}
print("   static perfect isolators with n-2 internal modes (cascaded chains, best found):")
for n_, (rt, dt_) in chains.items():
    A_, t_, rev = sup_last_first(casc(rt, dt_))
    print(f"     n={n_}: sup|A_out<-in| = {A_:.5f} (P = {A_**2:.4f}) at Gamma t = {t_:.3f}; reverse = {rev:.1e}")
A_, t_, _ = sup_last_first(casc([1, 1, 1], [0, 0, 0]))
print(f"     resonant equal-rate chain n=3: {A_:.5f} at t = {t_:.3f}")

# ======================================================================
header("S8  Channel-level reciprocity (Theorem 3(b),(c)) and its limits (Proposition 1)")

def gaussian_channel(Mseg, taus, gam, nbath, n0, i, j):
    """dP/dt = i conj(M) P - i P M^T + diag(gam*nbath), P_jk = <a_j^dag a_k>.
    Returns transmissivity |A_ji|^2 and added noise in mode j (input mode i excluded)."""
    n_ = Mseg[0].shape[0]; U = np.eye(n_, dtype=complex)
    P = np.diag(n0).astype(complex); P[i, i] = 0.0
    D = np.diag(gam * nbath)
    for M, tau in zip(Mseg, taus):
        A_ = 1j * np.kron(np.eye(n_), M.conj()) - 1j * np.kron(M, np.eye(n_))
        big = np.zeros((n_ * n_ + 1, n_ * n_ + 1), complex)
        big[:n_ * n_, :n_ * n_] = A_; big[:n_ * n_, n_ * n_] = D.reshape(-1, order="F")
        v = expm(big * tau) @ np.concatenate([P.reshape(-1, order="F"), [1]])
        P = v[:n_ * n_].reshape(n_, n_, order="F"); U = expm(-1j * M * tau) @ U
    return abs(U[j, i])**2, P[j, j].real

Hn = np.array([[0, 0, 1.0], [0, 0, 0.4], [1.0, 0.4, 0.3]], complex)
gam3 = np.array([0.05, 0.2, 0.8]); Mn = Hn - 0.5j * np.diag(gam3)
print("   two magnons + resonator, g=(1,0.4), Delta_q=0.3, gamma=(0.05,0.2,0.8), t=2.3:")
for lab, nb, n0 in [("common occupation 0.5", np.full(3, .5), np.full(3, .5)),
                    ("all at zero temperature", np.zeros(3), np.zeros(3)),
                    ("hot mediator bath n_q=2", np.array([0, 0, 2.]), np.zeros(3)),
                    ("hot magnon-1 bath n_1=1", np.array([1., 0, 0]), np.zeros(3))]:
    T12, N12 = gaussian_channel([Mn], [2.3], gam3, nb, n0, 0, 1)
    T21, N21 = gaussian_channel([Mn], [2.3], gam3, nb, n0, 1, 0)
    print(f"     {lab:26s}: T(1->2)={T12:.6f} T(2->1)={T21:.6f} | noise(1->2)={N12:.6f} noise(2->1)={N21:.6f}")
    if lab.startswith("common") or lab.startswith("all"):
        assert abs(T12 - T21) < 1e-12 and abs(N12 - N21) < 1e-12 and abs(N12 - nb[0] * (1 - T12)) < 1e-12

# time-dependent gauge: each segment is gauge-symmetric (a tree), palindromic protocol, yet non-reciprocal
def tri(c12, c23, c13, loss):
    H = np.array([[0, c12, c13], [np.conj(c12), 0, c23], [np.conj(c13), np.conj(c23), 0]], complex)
    return H - 0.5j * np.diag(loss)
lossv = [0.05, 0.05, 0.4]
MA = tri(np.exp(1j * pi / 4), 1.0, 0, lossv); MB = tri(0, 0, 1.0, lossv)
U = expm(-1j * MA * 0.8) @ expm(-1j * MB * 1.1) @ expm(-1j * MA * 0.8)
print(f"   time-dependent gauge (segments A-B-A, each a tree): |A21|^2={abs(U[1, 0])**2:.6f}  |A12|^2={abs(U[0, 1])**2:.6f}")
MA0 = tri(1.0, 1.0, 0, lossv)
U0 = expm(-1j * MA0 * 0.8) @ expm(-1j * MB * 1.1) @ expm(-1j * MA0 * 0.8)
print(f"   same with the phase removed (common gauge):        |A21|^2={abs(U0[1, 0])**2:.6f}  |A12|^2={abs(U0[0, 1])**2:.6f}")

# nonlinear (two-level) mediator versus linear resonator: two-excitation transfer, static, real couplings
def destroy(n):
    return np.diag(np.sqrt(np.arange(1, n)), 1)
Nf = 3; a = destroy(Nf); I3 = np.eye(Nf)
for med, dm in [("transmon (two-level)", 2), ("resonator", 3)]:
    sm = destroy(dm)
    m1 = np.kron(np.kron(a, I3), np.eye(dm)); m2 = np.kron(np.kron(I3, a), np.eye(dm))
    sq = np.kron(np.kron(I3, I3), sm)
    H = 0.4 * sq.conj().T @ sq + 1.0 * (m1.conj().T @ sq + sq.conj().T @ m1) + 0.5 * (m2.conj().T @ sq + sq.conj().T @ m2)
    Ls = [sqrt(0.05) * m1, sqrt(0.2) * m2, sqrt(0.6) * sq]
    dd = H.shape[0]; Id = np.eye(dd)
    Lv = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
    for C in Ls:
        CdC = C.conj().T @ C
        Lv += np.kron(C.conj(), C) - 0.5 * np.kron(Id, CdC) - 0.5 * np.kron(CdC.T, Id)
    E = expm(Lv * 3.0)
    def ket(n1, n2):
        v = np.zeros(dd); v[(n1 * Nf + n2) * dm] = 1; return v
    def run(psi):
        return (E @ np.outer(psi, psi).reshape(-1, order="F")).reshape(dd, dd, order="F")
    def marg(rho, op, n_):
        w_, V_ = np.linalg.eigh(op); P_ = V_[:, np.isclose(w_, n_)]
        return np.real(np.trace(P_.conj().T @ rho @ P_))
    r12, r21 = run(ket(2, 0)), run(ket(0, 2))
    n1op, n2op = m1.conj().T @ m1, m2.conj().T @ m2
    nf, nr = (ket(0, 2) @ r12 @ ket(0, 2)).real, (ket(2, 0) @ r21 @ ket(2, 0)).real
    pf, pr = marg(r12, n2op, 1), marg(r21, n1op, 1)
    print(f"   {med:21s}: P(|2,0>->|0,2>) fwd={nf:.6f} rev={nr:.6f} | P(exactly one arrives) fwd={pf:.6f} rev={pr:.6f}")
    assert abs(nf - nr) < 1e-10
print("\nAll assertions passed.")
