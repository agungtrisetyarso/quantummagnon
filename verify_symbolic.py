#!/usr/bin/env python3
"""Symbolic (SymPy) verification of the analytic steps in the manuscript.
Each check prints PASS or raises AssertionError. Runtime: a few seconds."""
import sympy as sp

def ok(name, cond):
    assert cond, name
    print(f"PASS  {name}")

t, u, s, k, k1, k2, v0, tau = sp.symbols("t u s k k1 k2 v0 tau", positive=True)
G, Gam, j, y, r, rho = sp.symbols("G Gamma j y r rho", positive=True)
a, b, eps = sp.symbols("a b epsilon", positive=True)

# ---- Theorem 4(a): cascaded form reproduces K and the isolation condition -----
kap1, kap2, phi = sp.symbols("kappa1 kappa2 phi", positive=True)
c = sp.Matrix([sp.sqrt(kap1), sp.exp(sp.I * phi) * sp.sqrt(kap2)])     # L = c1 m1 + c2 m2
K = sp.Matrix(2, 2, lambda p, q: sp.conjugate(c[p]) * c[q])          # K_pq = conj(c_p) c_q
H12 = sp.I / 2 * sp.sqrt(kap1 * kap2) * sp.exp(sp.I * phi)            # coefficient of m1^dag m2
ok("Thm 4(a): M12 = H12 - (i/2) K12 = 0", sp.simplify(H12 - sp.I / 2 * K[0, 1]) == 0)
M21 = sp.conjugate(H12) - sp.I / 2 * K[1, 0]
ok("Thm 4(a): M21 = -i K21", sp.simplify(M21 + sp.I * K[1, 0]) == 0)

# ---- Theorem 4(b): envelope integral and its representation ------------------
integrand = sp.exp(-(k1 * u + k2 * (t - u)) / 2).subs({k1: k * sp.exp(v0), k2: k * sp.exp(-v0)})
Bv = (k * sp.integrate(integrand, (u, 0, t), conds="none")).subs(t, 2 * tau / k)
target = (sp.exp(-tau * sp.exp(-v0)) - sp.exp(-tau * sp.exp(v0))) / sp.sinh(v0)
ok("Thm 4(b): B(t) closed form", all(abs(float(sp.N((Bv - target).subs({tau: tv, v0: vv_, k: kv})))) < 1e-14
                                   for tv in [0.3, 1, 2.5] for vv_ in [0.1, 0.7, 2.0] for kv in [0.5, 1, 3]))
vv = sp.symbols("v", real=True)
integral = sp.integrate(tau * sp.exp(vv) * sp.exp(-tau * sp.exp(vv)), (vv, -v0, v0))
ok("Thm 4(b): integral representation", sp.simplify(integral - (sp.exp(-tau * sp.exp(-v0)) - sp.exp(-tau * sp.exp(v0)))) == 0)
xx = sp.symbols("x", positive=True)
ok("Thm 4(b): max_x x e^{-x} = 1/e at x=1",
   sp.solve(sp.diff(xx * sp.exp(-xx), xx), xx) == [1] and sp.simplify((xx * sp.exp(-xx)).subs(xx, 1) - sp.exp(-1)) == 0)
ok("Thm 4(b): v0 = 0 limit is 2 tau e^{-tau}", sp.simplify(sp.limit(target, v0, 0) - 2 * tau * sp.exp(-tau)) == 0)

# ---- Theorem 4(c): exact supremum for unequal rates --------------------------
f = 2 * r * sp.sqrt(rho) * (sp.exp(-rho * t / 2) - sp.exp(-t / 2)) / (1 - rho)       # k1 = 1, k2 = rho
tstar = 2 * sp.log(rho) / (rho - 1)
dfs = sp.diff(f, t).subs(t, tstar)
ok("Thm 4(c): stationary point", all(abs(sp.N(dfs.subs({rho: rv, r: 1}), 40)) < 1e-35
                                   for rv in [sp.Rational(1, 3), 2, 4, 16, sp.Rational(11, 10)]))
val = f.subs(t, tstar)
claimed = 2 * r * rho ** ((1 + rho) / (2 * (1 - rho)))
for rv in [sp.Rational(1, 3), 2, 4, 16]:
    ok(f"Thm 4(c): value at rho={rv}", abs(sp.N((val - claimed).subs({rho: rv, r: 1}), 40)) < 1e-35)
ok("Thm 4(c): rho -> 1 limit = 2r/e", sp.simplify(sp.limit(claimed, rho, 1) - 2 * r / sp.E) == 0)

# ---- Dephasing extension: coherence and population equations ------------------
# with M12 = 0 the single-excitation density-matrix equations close as stated in the proof
M11, M21s, M22 = sp.symbols("M11 M21 M22")
R = sp.Matrix(2, 2, sp.symbols("r11 r12 r21 r22"))
Mm = sp.Matrix([[M11, 0], [M21s, M22]])
gp1, gp2 = sp.symbols("gphi1 gphi2", positive=True)
P1, P2 = sp.diag(1, 0), sp.diag(0, 1)
drho = -sp.I * (Mm * R - R * Mm.H)
for g_, P in [(gp1, P1), (gp2, P2)]:
    drho += g_ * (P * R * P - (P * R + R * P) / 2)
ok("Dephasing: d r11/dt involves r11 only", sp.simplify(drho[0, 0] - (-sp.I * (M11 - sp.conjugate(M11)) * R[0, 0])) == 0)
ok("Dephasing: d r21/dt has no r12, r22 terms", all(sp.diff(drho[1, 0], z) == 0 for z in [R[0, 1], R[1, 1]]))
ok("Dephasing: d r22/dt has no dephasing term", all(sp.diff(drho[1, 1], z) == 0 for z in [gp1, gp2]))

uu, ss = sp.symbols("uu ss", positive=True)
Fexpr = sp.integrate(sp.exp(-(k1 * uu + k2 * (t - uu)) / 2), (uu, 0, t), conds="none")
Rinner = sp.integrate(sp.exp(-(k1 + k2) * (t - ss) / 2 - k1 * ss), (ss, 0, t), conds="none")
ok("Thm 4(d): inner integral R(t) = e^{-k1 t/2} F(t)",
   all(abs(float(sp.N((Rinner - sp.exp(-k1 * t / 2) * Fexpr).subs({k1: a1, k2: a2, t: tt})))) < 1e-14
       for a1, a2 in [(1, 2), (0.3, 1.7), (2, 0.5)] for tt in [0.4, 1.3, 3.0]))
Gexpr = 2 * sp.integrate(sp.exp(-k2 * (t - uu)) * Rinner.subs(t, uu), (uu, 0, t), conds="none")
ok("Thm 4(d): 2 int e^{-k2(t-u)} R(u) du = F(t)^2",
   all(abs(float(sp.N((Gexpr - Fexpr ** 2).subs({k1: a1, k2: a2, t: tt})))) < 1e-13
       for a1, a2 in [(1, 2), (0.3, 1.7), (2, 0.5)] for tt in [0.4, 1.3, 3.0]))

# ---- Proposition 2: propagator of M_j and maximizer ----------------------------
w = sp.symbols("omega", positive=True)
N = sp.Matrix([[0, sp.I * Gam / 2 * (j - 1)], [-sp.I * Gam / 2 * (j + 1), 0]])
Mj = -sp.I * Gam / 2 * sp.ones(2, 2) + sp.Matrix([[0, sp.I * j * Gam / 2], [-sp.I * j * Gam / 2, 0]])
ok("Prop 2: M_j = -i Gamma/2 * 1 + N", sp.simplify(Mj - (-sp.I * Gam / 2 * sp.eye(2) + N)) == sp.zeros(2))
ok("Prop 2: N^2 = (Gamma/2)^2 (j^2-1) 1", sp.simplify(N * N - (Gam / 2) ** 2 * (j ** 2 - 1) * sp.eye(2)) == sp.zeros(2))
wj = Gam / 2 * sp.sqrt(j ** 2 - 1)
Ut = sp.exp(-Gam * t / 2) * (sp.cos(wj * t) * sp.eye(2) - sp.I * N * sp.sin(wj * t) / wj)
ok("Prop 2: U(t) solves dU/dt = -i M_j U", sp.simplify(sp.diff(Ut, t) + sp.I * Mj * Ut) == sp.zeros(2))
ok("Prop 2: U(0) = 1", sp.simplify(Ut.subs(t, 0)) == sp.eye(2))
h = sp.exp(-s) * sp.sin(y * s) / y
th = sp.atan(y) / y
ok("Prop 2: first maximum at s = arctan(y)/y", sp.simplify(sp.diff(h, s).subs(s, th)) == 0)
ok("Prop 2: value e^{-theta}/sqrt(1+y^2)", sp.simplify(h.subs(s, th) - sp.exp(-th) / sp.sqrt(1 + y ** 2)) == 0)

# ---- Theorem 2: Sylvester propagator for the bright/mediator block --------------
gm, gq, Dq = sp.symbols("gamma_m gamma_q Delta_q", positive=True)
lp, lm = sp.symbols("lambda_p lambda_m")
Mbs = sp.Matrix([[-sp.I * gm / 2, G], [G, Dq - sp.I * gq / 2]])
evs = list(Mbs.eigenvals().keys())
subs = {lp: evs[0], lm: evs[1]}
beta = ((Mbs[0, 0] - lm) * sp.exp(-sp.I * lp * t) - (Mbs[0, 0] - lp) * sp.exp(-sp.I * lm * t)) / (lp - lm)
q = G * (sp.exp(-sp.I * lp * t) - sp.exp(-sp.I * lm * t)) / (lp - lm)
vec = sp.Matrix([beta, q]).subs(subs)
res = sp.diff(vec, t) + sp.I * Mbs * vec
num = {gm: sp.Rational(1, 7), gq: sp.Rational(3, 2), Dq: 5, G: sp.Rational(6, 5), t: sp.Rational(13, 10)}
ok("Thm 2: (beta, q) solves the bright/mediator ODE", all(abs(complex(sp.N(z.subs(num), 30))) < 1e-25 for z in res))
ok("Thm 2: initial condition (1, 0)", sp.simplify(vec.subs(t, 0) - sp.Matrix([1, 0])) == sp.zeros(2, 1))

# ---- Corollary 1: asymptotics and the optimal detuning --------------------------
s_ = sp.exp(-b) * (1 + sp.exp(-a)) / 2          # s = xi^2 (1 + eta^2)/2 ; p0 = 1 - s ; 2|a1 a2| = s
p0 = 1 - s_
EN = sp.log(1 + sp.sqrt(p0 ** 2 + s_ ** 2) - p0, 2)
ser = sp.series(EN.subs({a: eps * a, b: eps * b}), eps, 0, 2).removeO()
ok("Cor 1: 1 - E_N = (b + a/2)/ln2 + O(2)", sp.simplify(1 - ser - eps * (b + a / 2) / sp.log(2)) == 0)
g_, gq_, gm_, D = sp.symbols("g gamma_q gamma_m Delta", positive=True)
a_expr = sp.pi * gq_ / (2 * D)                  # Purcell exponent at t* = pi D/(4 g^2)
b_expr = sp.pi * gm_ * D / (4 * g_ ** 2)        # magnon exponent at t*
crit = sp.solve(sp.diff(b_expr + a_expr / 2, D), D)
ok("Cor 1: optimal detuning D = g sqrt(gq/gm)", len(crit) == 1 and sp.simplify(crit[0] - g_ * sp.sqrt(gq_ / gm_)) == 0)
Cc = sp.symbols("C", positive=True)
deficit = sp.simplify(((b_expr + a_expr / 2) / sp.log(2)).subs(D, crit[0]).subs(gq_, g_ ** 2 / (Cc * gm_)))
ok("Cor 1: deficit = pi/(2 ln2) C^{-1/2}", sp.simplify(deficit - sp.pi / (2 * sp.log(2) * sp.sqrt(Cc))) == 0)
old = sp.simplify(((b_expr + a_expr / 2) / sp.log(2)).subs(D, g_ * sp.sqrt(2 * gq_ / gm_)).subs(gq_, g_ ** 2 / (Cc * gm_)))
ok("Cor 1: at D = g sqrt(2gq/gm) the constant is 3pi/(4 sqrt2 ln2) (suboptimal)",
   sp.simplify(old - 3 * sp.pi / (4 * sp.sqrt(2) * sp.log(2) * sp.sqrt(Cc))) == 0)
print(f"      pi/(2 ln 2) = {float(sp.pi / (2 * sp.log(2))):.6f};  3pi/(4 sqrt2 ln2) = {float(3 * sp.pi / (4 * sp.sqrt(2) * sp.log(2))):.6f}")

# ---- Lemma 4: (a) => (b) ---------------------------------------------------------
X = sp.Matrix(2, 2, sp.symbols("x0:4", real=True)); Y = sp.Matrix(2, 2, sp.symbols("y0:4", real=True))
Hh = sp.Matrix([[X[0, 0], X[0, 1] + sp.I * Y[0, 1]], [X[0, 1] - sp.I * Y[0, 1], X[1, 1]]])
Kk = sp.Matrix([[Y[0, 0], X[1, 0] + sp.I * Y[1, 0]], [X[1, 0] - sp.I * Y[1, 0], Y[1, 1]]])
Mx = Hh - sp.I / 2 * Kk
sol = sp.solve(list(Mx - Mx.T), [Y[0, 1], Y[1, 0]], dict=True)
ok("Lemma 4: M symmetric forces Im H12 = Im K12 = 0", sol == [{Y[0, 1]: 0, Y[1, 0]: 0}])
print("\nAll symbolic checks passed.")
