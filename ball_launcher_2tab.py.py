"""20-ft ball launcher: aiming + launcher design, two-tab app (single file).

Tab 1  Trajectory & Aiming
       Pick a ball size; the program re-aims to keep landing at 20 ft.
         1) change the angle (low-angle solution only, never below 31 deg)
         2) if 31 deg still lands long, stay at 31 deg and turn the stopper in
       31 deg is a hard limit: below it the ball rolls off the 60 deg piston seat.
       Ball-size plan (angle + stopper per ball) lives here too.

Tab 2  Launcher Design
       Current spring (from its geometry), stopper, piston-side mass.
       - speed vs piston-side mass with the current spring / stopper
       - energy margin verdict: PASS (>= target) / MARGINAL / FAIL
       - if not PASS: A) same spring, deeper latch   B) new spring (searched)
                      C) max piston-side mass the current spring can carry
       A what-if mass analyses a scenario without changing tab 1.

Launcher model (energy balance latch -> stop; ball leaves when piston hits the stop)
    1/2 k (x_L^2 - x_P^2) = (1/eta) [ 1/2 m_eff v0^2 + m_lift g s sin(theta) ],  s = x_L - x_P
    x_L = spring compression at latch, x_P = preload at stop (stopper dial),
    m_move = ball + piston-side hardware
    m_eff  = m_move + m_spring/3   (kinetic: coil velocity linear along the spring, KE ~ v^2)
    m_lift = m_move + m_spring/2   (gravity: spring centre of mass rises s/2)
Flight: point mass, quadratic air drag (always on), RK4.
Spring: Shigley - Bergstrasser factor, Sut = A/d^m, static allowable 0.45 Sut,
        Zimmerli fatigue data (peened / unpeened) with Goodman, fn fixed-fixed.

Units: ft, ft/s, lbm, lbf; launcher inputs in in, lbf/in, ksi.
Run: python ball_launcher_2tab.py              GUI (Tkinter + matplotlib)
     python ball_launcher_2tab.py --self-test
     python ball_launcher_2tab.py --demo       writes tab PNGs + ball-size plan CSV
"""
from __future__ import annotations
import math
import sys
from dataclasses import dataclass, replace
from functools import lru_cache

G = 32.1740485564        # ft/s^2
G_IN = 386.0886          # in/s^2
TARGET = 20.0            # ft
LBM_PER_SLUG = 32.1740485564
RHO = 0.002377           # slug/ft^3
MASS_BALL_LBM = 1.0

# ---- shot ---------------------------------------------------------------------
ANGLE_NOMINAL_DEG = 37.0
ANGLE_MIN_DEG = 31.0      # HARD LIMIT: below 31 deg the ball rolls off the 60 deg piston seat
ANGLE_MAX_DEG = 50.0
HEIGHT_FT = 5.0
DIAM_DEFAULT_IN = 4.0
DIAM_MIN_IN = 1.9
DIAM_MAX_IN = 4.75
CD = 0.47
TOL_FT = 0.5
WIND_CHECK_MPH = 2.0

# ---- current spring (custom Cr-Si, shot-peened) -------------------------------
MATERIAL = 'Cr-Si A401'
PEENED = True
WIRE_D = 0.187
COIL_D = 2.24             # mean diameter
N_ACTIVE = 12.2
N_TOTAL = 14.2            # closed & ground
FREE_LEN = 10.1
X_COCK_IN = 6.5           # compression at latch (latch height 3.6 in)
# ---- stopper -------------------------------------------------------------------
P_MIN_IN = 0.5
P_MAX_IN = 2.7
THREAD_TPI = 20.0         # 1/2-20 UNF
# ---- piston side (spring/3 is added automatically) ----------------------------
M_PISTON_ASSY = 0.547     # piston 0.51 (CAD) + 3/8-16 nylon locknut 0.017 + 1 in USS washer 0.015 + soft loop 0.005
M_STOPPER = 0.15          # stopper collar + jam nut (CAD)
ROD_LEN_IN = 17.14         # equivalent full-Ø0.5 length for the CAD mass 0.34 lb (actual rod 18 in, ends turned down)
ROD_D_IN = 0.5
ROD_MATERIAL = '7075-T6 Al'
ETA = 0.985               # ASSUMED; calibrate with test shots
# ---- design criteria -----------------------------------------------------------
MARGIN_TARGET = 0.0       # requirement: the worst-case ball reaches 20 ft (margin is reported, not forced)
SOLID_CLEAR_FRAC = 0.15   # gap between latch and solid, as a fraction of latch compression x_L
SF_SOLID_MIN = 1.2
SF_GOODMAN_MIN = 1.5
OD_MAX = 2.5              # piston seat
ID_MIN = 1.0
F_COCK_MAX = 100.0        # lbf at latch (2:1 cocker -> half at the hand)
L0_MAX_IN = 12.1          # longest free length allowed for a new spring
L0_ROUND = 0.1            # free length is specified to 0.1 in, so allow half of that on the clearance check
BUCKLE_RATIO = 5.26       # L0/D limit, ends on parallel flat plates (alpha = 0.5)

MATERIALS = {'Cr-Si A401': dict(G=11.2e6, A=261.0, m=0.108, rho=0.284),
             'Music wire A228': dict(G=11.75e6, A=201.0, m=0.145, rho=0.284)}
ROD_DENSITY = {'7075-T6 Al': 0.101, '6061-T6 Al': 0.098, 'Ti-6Al-4V': 0.160, 'Steel': 0.284}
ZIMMERLI = {True: (57.5, 77.5), False: (35.0, 55.0)}   # Ssa, Ssm  kpsi
SSY_FRAC = 0.45
WIRE_SIZES = (0.148, 0.156, 0.162, 0.170, 0.177, 0.180, 0.187, 0.192, 0.200, 0.207,
              0.218, 0.225, 0.234, 0.243, 0.250, 0.262)


@dataclass(frozen=True)
class Config:
    diameter_in: float = DIAM_DEFAULT_IN
    height_ft: float = HEIGHT_FT
    cd: float = CD
    angle_min: float = ANGLE_MIN_DEG
    angle_max: float = ANGLE_MAX_DEG
    angle_nominal: float = ANGLE_NOMINAL_DEG
    material: str = MATERIAL
    peened: bool = PEENED
    wire_d: float = WIRE_D
    coil_d: float = COIL_D
    n_active: float = N_ACTIVE
    n_total: float = N_TOTAL
    free_len: float = FREE_LEN
    x_cock_in: float = X_COCK_IN
    p_min_in: float = P_MIN_IN
    p_max_in: float = P_MAX_IN
    tpi: float = THREAD_TPI
    m_piston_assy: float = M_PISTON_ASSY
    m_stopper: float = M_STOPPER
    rod_len_in: float = ROD_LEN_IN
    rod_material: str = ROD_MATERIAL
    eta: float = ETA
    baseline_ball_in: float = DIAM_MAX_IN
    stopper_turns: float | None = None     # None = auto baseline; else manual turns
    margin_target: float = MARGIN_TARGET
    solid_clear: float = SOLID_CLEAR_FRAC
    od_max: float = OD_MAX
    id_min: float = ID_MIN
    f_cock_max: float = F_COCK_MAX
    l0_max: float = L0_MAX_IN


# ================================================================ flight ====
def _hermite(p0, p1, m0, m1, s):
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1


def _hermite_root(y0, y1, m0, m1):
    lo, hi = 0., 1.
    s_ = y0 / (y0 - y1) if y0 != y1 else 1.
    for _ in range(8):
        f = _hermite(y0, y1, m0, m1, s_)
        df = (6 * s_ * s_ - 6 * s_) * y0 + (3 * s_ * s_ - 4 * s_ + 1) * m0 + (-6 * s_ * s_ + 6 * s_) * y1 + (3 * s_ * s_ - 2 * s_) * m1
        if f > 0:
            lo = s_
        else:
            hi = s_
        n = s_ - f / df if df else (lo + hi) / 2
        s_ = n if lo < n < hi else (lo + hi) / 2
        if abs(f) < 1e-12:
            break
    return s_


def trajectory(height_ft, diameter_in, cd, angle_deg, speed, samples=True, headwind=0.0):
    """RK4 flight with quadratic drag. samples=False: dt 0.02 s + Hermite touchdown
    (landing error < 1e-6 in); samples=True: dt 0.005 s for plotting. Returns (xs, ys, t)."""
    th = math.radians(angle_deg)
    vx, vy = speed * math.cos(th), speed * math.sin(th)
    x, y = 0., height_ft
    kd = 0.5 * RHO * cd * math.pi * (diameter_in / 24.0) ** 2 / (MASS_BALL_LBM / LBM_PER_SLUG)
    dt = 0.005 if samples else 0.02
    h2, h6 = dt / 2, dt / 6
    hyp = math.hypot
    xs, ys = ([x], [y]) if samples else (None, None)
    t = 0.
    for _ in range(20000):
        r = hyp(vx + headwind, vy); a1x = -kd * r * (vx + headwind); a1y = -G - kd * r * vy
        v2x, v2y = vx + h2 * a1x, vy + h2 * a1y
        r = hyp(v2x + headwind, v2y); a2x = -kd * r * (v2x + headwind); a2y = -G - kd * r * v2y
        v3x, v3y = vx + h2 * a2x, vy + h2 * a2y
        r = hyp(v3x + headwind, v3y); a3x = -kd * r * (v3x + headwind); a3y = -G - kd * r * v3y
        v4x, v4y = vx + dt * a3x, vy + dt * a3y
        r = hyp(v4x + headwind, v4y); a4x = -kd * r * (v4x + headwind); a4y = -G - kd * r * v4y
        nx = x + h6 * (vx + 2 * v2x + 2 * v3x + v4x)
        ny = y + h6 * (vy + 2 * v2y + 2 * v3y + v4y)
        nvx = vx + h6 * (a1x + 2 * a2x + 2 * a3x + a4x)
        nvy = vy + h6 * (a1y + 2 * a2y + 2 * a3y + a4y)
        if ny <= 0:
            s_ = _hermite_root(y, ny, vy * dt, nvy * dt)
            xl = _hermite(x, nx, vx * dt, nvx * dt, s_)
            if samples:
                xs.append(xl); ys.append(0.)
                return xs, ys, t + s_ * dt
            return [xl], [0.], t + s_ * dt
        x, y, vx, vy = nx, ny, nvx, nvy
        t += dt
        if samples:
            xs.append(x); ys.append(y)
    raise ValueError('Flight integration exceeded limits')


def landing_at(cfg: Config, angle, speed):
    if speed <= 0:
        return 0.0
    return trajectory(cfg.height_ft, cfg.diameter_in, cfg.cd, angle, speed, samples=False)[0][-1]


@lru_cache(maxsize=512)
def _req_speed(height, diam, cd, angle):
    th = math.radians(angle)
    v = math.sqrt(G * TARGET ** 2 / (2 * math.cos(th) ** 2 * (height + TARGET * math.tan(th))))
    for _ in range(30):
        r = trajectory(height, diam, cd, angle, v, samples=False)[0][-1]
        if abs(r - TARGET) < 1e-6:
            break
        v *= math.sqrt(TARGET / r)
    return v


def required_speed(cfg: Config, angle):
    """Release speed that lands at exactly 20 ft (with drag) at this angle."""
    return _req_speed(cfg.height_ft, cfg.diameter_in, cfg.cd, angle)


def worst_speed(cfg: Config):
    """Design point: worst-case (largest) ball at the nominal angle."""
    return _req_speed(cfg.height_ft, cfg.baseline_ball_in, cfg.cd, cfg.angle_nominal)


# ================================================================ spring ====
def spring_k(c: Config):
    return MATERIALS[c.material]['G'] * c.wire_d ** 4 / (8 * c.coil_d ** 3 * c.n_active)


def spring_mass(c: Config):
    return MATERIALS[c.material]['rho'] * math.pi ** 2 * c.wire_d ** 2 * c.coil_d * c.n_total / 4


def solid_len(c: Config):
    return c.n_total * c.wire_d


def rod_mass(c: Config):
    return ROD_DENSITY[c.rod_material] * math.pi / 4 * ROD_D_IN ** 2 * c.rod_len_in


def m_hardware(c: Config):
    """Piston-side hardware without the spring."""
    return c.m_piston_assy + c.m_stopper + rod_mass(c)


def m_move(c: Config, m_hw=None):
    """Ball + piston-side hardware (no spring)."""
    return MASS_BALL_LBM + (m_hardware(c) if m_hw is None else m_hw)


def moving_mass(c: Config, m_hw=None):
    """m_eff for kinetic energy: m_move + m_spring/3 (velocity varies linearly along the coil, KE ~ v^2)."""
    return m_move(c, m_hw) + spring_mass(c) / 3


def lift_mass(c: Config, m_hw=None):
    """Mass for the gravity (lift) term: m_move + m_spring/2 (spring centre of mass moves s/2)."""
    return m_move(c, m_hw) + spring_mass(c) / 2


def spring_energy(c: Config, p, x_c=None):
    x_c = c.x_cock_in if x_c is None else x_c
    return 0.5 * spring_k(c) * (x_c ** 2 - p ** 2)


def launch_speed(c: Config, angle, p, m_hw=None):
    m = moving_mass(c, m_hw)
    net = c.eta * spring_energy(c, p) - lift_mass(c, m_hw) * math.sin(math.radians(angle)) * (c.x_cock_in - p)
    return math.sqrt(2 * net / 12.0 / (m / LBM_PER_SLUG)) if net > 0 else 0.0


def preload_for_speed(c: Config, angle, v, m_hw=None):
    """Inverse of launch_speed. NaN if not reachable even at p = 0."""
    m = moving_mass(c, m_hw)
    ws = lift_mass(c, m_hw) * math.sin(math.radians(angle))
    ke = 0.5 * (m / LBM_PER_SLUG) * v * v * 12.0
    a = c.eta * spring_k(c) / 2
    cc = a * c.x_cock_in ** 2 - ws * c.x_cock_in - ke
    disc = ws * ws + 4 * a * cc
    return (ws + math.sqrt(disc)) / (2 * a) if disc >= 0 else float('nan')


def energy_needed(c: Config, v, p, m_hw=None, x_c=None):
    """Spring energy needed (before losses) for release speed v from latch x_c to stop p."""
    x_c = c.x_cock_in if x_c is None else x_c
    m = moving_mass(c, m_hw)
    return (0.5 * (m / LBM_PER_SLUG) * v * v * 12.0
            + lift_mass(c, m_hw) * math.sin(math.radians(c.angle_nominal)) * (x_c - p)) / c.eta


def margin(c: Config, m_hw=None):
    """Energy margin with the stopper fully open, worst-case ball at the nominal angle."""
    return spring_energy(c, c.p_min_in) / energy_needed(c, worst_speed(c), c.p_min_in, m_hw) - 1


def verdict(c: Config, m):
    return 'PASS' if m >= c.margin_target - 1e-9 else ('MARGINAL' if m >= 0 else 'FAIL')


def spring_check(c: Config, m_hw=None):
    """Geometry, loads, stresses and checks for the spring in c."""
    mat = MATERIALS[c.material]
    d, D = c.wire_d, c.coil_d
    k = spring_k(c)
    C = D / d
    KB = (4 * C + 2) / (4 * C - 3)
    Sut = mat['A'] / d ** mat['m']
    Ssy, Ssu = SSY_FRAC * Sut, 0.67 * Sut
    tpf = KB * 8 * D / (math.pi * d ** 3) / 1000          # ksi per lbf
    Ls = solid_len(c)
    travel = c.free_len - Ls
    F_latch, F_solid, F_min = k * c.x_cock_in, k * max(travel, 1e-9), k * c.p_min_in
    tau_latch, tau_solid = tpf * F_latch, tpf * F_solid
    Ssa, Ssm = ZIMMERLI[c.peened]
    Sse = Ssa / (1 - Ssm / Ssu)
    tau_a, tau_m = tpf * (F_latch - F_min) / 2, tpf * (F_latch + F_min) / 2
    n_good = 1 / (tau_a / Sse + tau_m / Ssu)
    clear = c.free_len - c.x_cock_in - Ls
    w = spring_mass(c)
    fn = 0.5 * math.sqrt(k * G_IN / w)
    M = moving_mass(c, m_hw)
    omega = math.sqrt(k * G_IN / M)                      # rad/s (M lbm, k lbf/in)
    x_eq = lift_mass(c, m_hw) * math.sin(math.radians(c.angle_nominal)) / k
    t_launch = math.acos(max(-1, min(1, (c.p_min_in - x_eq) / (c.x_cock_in - x_eq)))) / omega
    mg = margin(c, m_hw)
    chk = dict(
        solid=Ssy / tau_solid >= SF_SOLID_MIN,
        goodman=n_good >= SF_GOODMAN_MIN,
        clear=clear + L0_ROUND / 2 >= c.solid_clear * c.x_cock_in - 1e-9,   # % of x_L; free length is rounded to 0.1 in
        buckle=c.free_len / D <= BUCKLE_RATIO,
        od=D + d <= c.od_max + 1e-9,
        id=D - d >= c.id_min - 1e-9,
        force=F_latch <= c.f_cock_max + 1e-9,
        margin=mg >= c.margin_target - 1e-9)
    return dict(k=k, C=C, OD=D + d, ID=D - d, Ls=Ls, travel=travel, F_latch=F_latch, F_min=F_min,
                tau_latch=tau_latch, tau_solid=tau_solid, Sut=Sut, Ssy=Ssy,
                sf_solid=Ssy / tau_solid, sf_static=Ssy / tau_latch, n_goodman=n_good,
                clear=clear, clear_frac=clear / c.x_cock_in,
                buckle=c.free_len / D, mass=w, fn=fn, t_launch=t_launch, cycles=t_launch * fn,
                margin=mg, verdict=verdict(c, mg), chk=chk, cfg=c,
                e_max=spring_energy(c, c.p_min_in), e_min=spring_energy(c, c.p_max_in))


def fix_latch(c: Config, m_hw=None):
    """Option A: same spring, latch compression for the target margin."""
    k = spring_k(c)
    m = moving_mass(c, m_hw)
    t = 1 + c.margin_target
    ws = lift_mass(c, m_hw) * math.sin(math.radians(c.angle_nominal))
    ke = 0.5 * (m / LBM_PER_SLUG) * worst_speed(c) ** 2 * 12.0
    p = c.p_min_in
    # 1/2 k x^2 - t ws/eta x - (1/2 k p^2 + t (ke - ws p)/eta) = 0
    a, b, cc = 0.5 * k, -t * ws / c.eta, -(0.5 * k * p * p + t * (ke - ws * p) / c.eta)
    x = (-b + math.sqrt(b * b - 4 * a * cc)) / (2 * a)
    return spring_check(replace(c, x_cock_in=x), m_hw)


def fix_spring(c: Config, m_hw=None, top=3):
    """Option B: new spring, same latch compression / stopper / material.
    Free length is derived, not chosen:  L0 = solid height + x_L * (1 + clearance)
    (solid height = N_t * d, closed & ground). The latch height becomes L0 - x_L.
    Keeps the current wire size if possible (minimum change), else searches all sizes;
    lightest spring that passes every check. Returns (passing list, tried, closest miss)."""
    mat = MATERIALS[c.material]
    t = 1 + c.margin_target
    span = 0.5 * (c.x_cock_in ** 2 - c.p_min_in ** 2)
    tried, closest, out = 0, None, []
    # minimum change first: keep the current wire size; open up all sizes only if nothing passes
    for wires in ((c.wire_d,), WIRE_SIZES):
      if out:
        break
      for d in wires:
          D = math.ceil(max(4 * d, c.id_min + d) * 100 - 1e-9) / 100
          while D + d <= c.od_max + 1e-9 and D / d <= 16:
              n_tot = c.n_total
              for _ in range(8):                          # spring mass <-> required rate
                  trial = replace(c, wire_d=d, coil_d=D, n_total=n_tot, n_active=n_tot - 2)
                  k_req = t * energy_needed(trial, worst_speed(c), c.p_min_in, m_hw) / span
                  na = mat['G'] * d ** 4 / (8 * D ** 3 * k_req)
                  n_tot = na + 2
              na = math.floor(na * 10) / 10                # round coils down -> rate up
              tried += 1
              if na >= 3:
                  L0 = math.ceil(((na + 2) * d + c.x_cock_in * (1 + c.solid_clear)) * 100 - 1e-9) / 100
                  if L0 <= c.l0_max + 1e-9:
                      sc = spring_check(replace(c, wire_d=d, coil_d=D, n_active=na, n_total=na + 2, free_len=L0), m_hw)
                      nfail = sum(not v for v in sc['chk'].values())
                      if nfail == 0:
                          out.append(sc)
                      elif closest is None or (nfail, sc['mass']) < (closest[0], closest[1]['mass']):
                          closest = (nfail, sc)
              D = round(D + 0.01, 4)
    out.sort(key=lambda q: q['mass'])
    return out[:top], tried, (None if out else (closest[1] if closest else None))


def comsol_check(k, x_cock, stroke, m_ball_h, m_spring_eff_h, m_added_h, m_piston_h, angle_deg):
    """Release speed for the COMSOL half-model inputs (masses in lb, half model -> x2).
    Same physics as COMSOL: no friction, every moving mass (incl. spring effective mass)
    is accelerated by the spring and lifted by gravity along the barrel.
        1/2 M v^2 = 1/2 k (x_cock^2 - x_P^2) - M g sin(theta) * stroke,  x_P = x_cock - stroke
    Returns v in ft/s."""
    M = 2 * (m_ball_h + m_spring_eff_h + m_added_h + m_piston_h)
    xp = x_cock - stroke
    e = 0.5 * k * (x_cock ** 2 - xp ** 2) - M * math.sin(math.radians(angle_deg)) * stroke   # in-lbf
    return math.sqrt(2 * e / 12.0 / (M / LBM_PER_SLUG)) if e > 0 else 0.0


def max_hw_mass(c: Config, target):
    """Largest piston-side hardware mass (no spring) that keeps margin >= target."""
    v = worst_speed(c)
    E = c.eta * spring_energy(c, c.p_min_in) / (1 + target)
    a = 0.5 * v * v * 12 / LBM_PER_SLUG                                  # per lb in the KE term
    b = math.sin(math.radians(c.angle_nominal)) * (c.x_cock_in - c.p_min_in)  # per lb in the lift term
    ws = spring_mass(c)
    return (E - ws * (a / 3 + b / 2)) / (a + b) - MASS_BALL_LBM


# ================================================================ stopper ===
def turns_of(cfg, p):
    return (p - cfg.p_min_in) * cfg.tpi


def p_of_turns(cfg, n):
    return cfg.p_min_in + n / cfg.tpi


def design_key(cfg: Config):
    return replace(cfg, diameter_in=cfg.baseline_ball_in)


def baseline_preload(cfg: Config):
    """Stopper baseline: worst-case ball at the nominal angle (or manual turns).
    Clamped to the stopper range: if the spring is too weak it sits fully open."""
    if cfg.stopper_turns is not None:
        return p_of_turns(cfg, cfg.stopper_turns)
    p = preload_for_speed(cfg, cfg.angle_nominal, worst_speed(cfg))
    if not math.isfinite(p) or p < cfg.p_min_in:
        return cfg.p_min_in
    return min(p, cfg.p_max_in)


def landing_with_stopper(cfg, angle, p):
    return landing_at(cfg, angle, launch_speed(cfg, angle, p))


# ================================================================ aiming ====
def _bisect(fun, a, b, tol=1e-4):
    fa, fb = fun(a), fun(b)
    if fa == 0:
        return a
    side = 0
    for _ in range(60):
        if b - a < tol:
            break
        m = (a * fb - b * fa) / (fb - fa) if fb != fa else (a + b) / 2
        if not a < m < b:
            m = (a + b) / 2
        fm = fun(m)
        if abs(fm) < 1e-6:
            return m
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
            if side == -1:
                fb /= 2
            side = -1
        else:
            b, fb = m, fm
            if side == 1:
                fa /= 2
            side = 1
    return (a + b) / 2


def peak_angle(cfg, p):
    lo, hi = cfg.angle_min, cfg.angle_max
    gr = (math.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = landing_with_stopper(cfg, c, p), landing_with_stopper(cfg, d, p)
    while hi - lo > 0.01:
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo); fc = landing_with_stopper(cfg, c, p)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo); fd = landing_with_stopper(cfg, d, p)
    a = (lo + hi) / 2
    return a, landing_with_stopper(cfg, a, p)


def cd_sensitivity(cfg, angle, v, frac=0.10):
    lo = landing_at(replace(cfg, cd=cfg.cd * (1 - frac)), angle, v)
    hi = landing_at(replace(cfg, cd=cfg.cd * (1 + frac)), angle, v)
    return (lo - TARGET) * 12, (hi - TARGET) * 12


def wind_sensitivity(cfg, angle, v, mph=WIND_CHECK_MPH):
    w = mph * 5280 / 3600
    r = trajectory(cfg.height_ft, cfg.diameter_in, cfg.cd, angle, v, samples=False, headwind=w)[0][-1]
    return (r - TARGET) * 12


def angle_sensitivity(cfg, angle, v):
    h = 0.25
    return (landing_at(cfg, angle + h, v) - landing_at(cfg, angle - h, v)) / (2 * h) * 12


def aim(cfg: Config):
    """Angle first, stopper second. ok = False if the design cannot reach 20 ft."""
    validate(cfg)
    p0 = baseline_preload(cfg)
    a_pk, r_pk = peak_angle(cfg, p0)
    f = lambda a: landing_with_stopper(cfg, a, p0) - TARGET
    mode, angle, p, branch, note = None, None, p0, None, ''
    if r_pk >= TARGET - 1e-3:
        if r_pk - TARGET <= 1e-3:
            angle, branch = a_pk, 'peak'
        elif f(cfg.angle_min) <= 0:
            angle, branch = _bisect(f, cfg.angle_min, a_pk), 'low'
        # no high-angle branch: if even the minimum angle (31 deg) lands long,
        # stay at the minimum angle and turn the stopper in (below)
        if angle is not None:
            mode = 'angle'
    if mode is None:
        if r_pk < TARGET:
            angle, branch = a_pk, 'peak'
            note = 'not enough energy at any angle: stopper moved back (longer stroke)'
        else:
            angle, branch = cfg.angle_min, 'low'
            note = 'too much energy at the angle limit: stopper moved forward (shorter stroke)'
        p = preload_for_speed(cfg, angle, required_speed(cfg, angle))
        mode = 'stopper'
    ok = math.isfinite(p) and cfg.p_min_in - 1e-9 <= p <= cfg.p_max_in + 1e-9
    if not ok:                                    # clamp to the stopper range and show the miss
        p = cfg.p_min_in if (not math.isfinite(p) or p < cfg.p_min_in) else cfg.p_max_in
        if p == cfg.p_min_in:
            angle, branch = peak_angle(cfg, p)[0], 'peak'
        note = 'stopper fully open and still short: the spring cannot reach 20 ft (see Design tab)' \
            if p == cfg.p_min_in else 'stopper at max preload and still long'
    v = launch_speed(cfg, angle, p)
    sol = dict(mode=mode, branch=branch, angle=angle, p=p, p0=p0, ok=ok, note=note, v=v,
               turns=turns_of(cfg, p), turns0=turns_of(cfg, p0), stroke=cfg.x_cock_in - p,
               a_peak=a_pk, r_peak=r_pk, landing=landing_at(cfg, angle, v))
    sol['high_warning'] = ok and branch == 'high'
    sol['cd_lo_in'], sol['cd_hi_in'] = cd_sensitivity(cfg, angle, v)
    sol['dRdth_in'] = angle_sensitivity(cfg, angle, v)
    sol['wind_in'] = wind_sensitivity(cfg, angle, v)
    sol['t_flight'] = trajectory(cfg.height_ft, cfg.diameter_in, cfg.cd, angle, v, samples=False)[2]
    if sol['high_warning']:
        pa = preload_for_speed(cfg, cfg.angle_min, required_speed(cfg, cfg.angle_min))
        va = launch_speed(cfg, cfg.angle_min, pa)
        sol['alt'] = dict(angle=cfg.angle_min, p=pa, turns=turns_of(cfg, pa),
                          ok=cfg.p_min_in <= pa <= cfg.p_max_in,
                          cd=cd_sensitivity(cfg, cfg.angle_min, va),
                          wind_in=wind_sensitivity(cfg, cfg.angle_min, va),
                          t_flight=trajectory(cfg.height_ft, cfg.diameter_in, cfg.cd, cfg.angle_min, va, samples=False)[2])
    return sol


def flight(cfg: Config, sol):
    xs, ys, t = trajectory(cfg.height_ft, cfg.diameter_in, cfg.cd, sol['angle'], sol['v'])
    i = max(range(len(ys)), key=ys.__getitem__)
    th = math.radians(sol['angle'])
    return dict(x=xs, y=ys, t=t, range=xs[-1], apex_x=xs[i], apex_y=ys[i],
                v_x=sol['v'] * math.cos(th), v_y=sol['v'] * math.sin(th),
                ke_ball=0.5 * (MASS_BALL_LBM / LBM_PER_SLUG) * sol['v'] ** 2 * 12.0)


def ball_table(cfg: Config, diams=(1.9, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75)):
    return _ball_table_cached(design_key(cfg), tuple(diams))


@lru_cache(maxsize=16)
def _ball_table_cached(key: Config, diams):
    return [(d, aim(replace(key, diameter_in=d))) for d in diams]


def warning_text(sol):
    if not sol['high_warning']:
        return ''
    t = (f"HIGH-ANGLE SOLUTION - LESS ACCURATE: flight {sol['t_flight']:.2f} s, so drag and wind act longer "
         f"({WIND_CHECK_MPH:g} mph headwind {sol['wind_in']:+.1f} in, Cd +10% {sol['cd_hi_in']:+.1f} in).")
    a = sol.get('alt')
    if a and a['ok']:
        t += (f" Low-angle option {a['angle']:.1f}° + stopper {a['turns']:+.1f} turns: "
              f"{a['t_flight']:.2f} s, wind {a['wind_in']:+.1f} in, Cd +10% {a['cd'][1]:+.1f} in.")
    return t


# ============================================================ design study ===
def design_study(cfg: Config, m_whatif=None):
    """Tab 2. m_whatif = piston-side hardware mass to analyse (None = design)."""
    validate(cfg)
    m_design = m_hardware(cfg)
    m_an = m_design if m_whatif is None else m_whatif
    p_cur = baseline_preload(cfg)                       # stopper as set for the design mass
    v_req = worst_speed(cfg)
    ref = replace(cfg, diameter_in=cfg.baseline_ball_in)
    cur = spring_check(cfg, m_an)
    v_cur = launch_speed(cfg, cfg.angle_nominal, p_cur, m_an)
    p_need = preload_for_speed(cfg, cfg.angle_nominal, v_req, m_an)
    reach = math.isfinite(p_need) and cfg.p_min_in <= p_need <= cfg.p_max_in
    A = B = B_close = None
    b_tried = 0
    if cur['verdict'] != 'PASS':
        A = fix_latch(cfg, m_an)
        Bs, b_tried, B_close = fix_spring(cfg, m_an)
        B = Bs[0] if Bs else None
    masses = [0.5 + 0.025 * i for i in range(73)]       # 0.5 .. 2.3 lb
    land = lambda c, m: landing_at(ref, cfg.angle_nominal, launch_speed(c, cfg.angle_nominal, c.p_min_in, m))
    sweep = dict(m=masses,
                 land_cur=[land(cfg, m) for m in masses],
                 land_B=[land(B['cfg'], m) for m in masses] if B else None,
                 v_cur=[launch_speed(cfg, cfg.angle_nominal, p_cur, m) for m in masses],
                 v_full=[launch_speed(cfg, cfg.angle_nominal, cfg.p_min_in, m) for m in masses],
                 mg_cur=[margin(cfg, m) for m in masses],
                 mg_A=[margin(A['cfg'], m) for m in masses] if A else None,
                 mg_B=[margin(B['cfg'], m) for m in masses] if B else None)
    return dict(m_design=m_design, m_an=m_an, whatif=m_whatif is not None, p_cur=p_cur,
                turns_cur=turns_of(cfg, p_cur), v_req=v_req, v_cur=v_cur,
                land_cur=landing_at(ref, cfg.angle_nominal, v_cur), p_need=p_need, reach=reach,
                turns_need=turns_of(cfg, p_need) if math.isfinite(p_need) else float('nan'),
                cur=cur, A=A, B=B, B_close=B_close, b_tried=b_tried,
                m_max_target=max_hw_mass(cfg, cfg.margin_target), m_max_zero=max_hw_mass(cfg, 0.0),
                m_max_zero_B=max_hw_mass(B['cfg'], 0.0) if B else None,
                land_full=land(cfg, m_an), land_full_B=land(B['cfg'], m_an) if B else None,
                eff=MASS_BALL_LBM / moving_mass(cfg, m_an), rod=rod_mass(cfg), sweep=sweep)


def validate(cfg: Config):
    nums = (cfg.diameter_in, cfg.height_ft, cfg.cd, cfg.angle_min, cfg.angle_max, cfg.angle_nominal,
            cfg.wire_d, cfg.coil_d, cfg.n_active, cfg.n_total, cfg.free_len, cfg.x_cock_in,
            cfg.p_min_in, cfg.p_max_in, cfg.tpi, cfg.m_piston_assy, cfg.m_stopper, cfg.rod_len_in,
            cfg.eta, cfg.baseline_ball_in, cfg.margin_target, cfg.solid_clear, cfg.od_max, cfg.id_min,
            cfg.f_cock_max)
    if not all(math.isfinite(v) for v in nums):
        raise ValueError('All inputs must be finite numbers')
    for d in (cfg.diameter_in, cfg.baseline_ball_in):
        if not DIAM_MIN_IN - 1e-9 <= d <= DIAM_MAX_IN + 1e-9:
            raise ValueError(f'Ball diameter must be {DIAM_MIN_IN}-{DIAM_MAX_IN} in (5.000 in bore / cup inserts)')
    if cfg.angle_min < ANGLE_MIN_DEG - 1e-9:
        raise ValueError(f'Angle min must be >= {ANGLE_MIN_DEG:g} deg (ball rolls off the piston seat below this)')
    if not cfg.angle_min < cfg.angle_max <= 85:
        raise ValueError(f'Angle range max must be > {cfg.angle_min:g} deg (the minimum) and <= 85 deg')
    # nominal angle = spring design point only (tab 2); it does not have to lie inside the aiming range
    if not ANGLE_MIN_DEG - 1e-9 <= cfg.angle_nominal <= 85:
        raise ValueError(f'Design-point nominal angle must be {ANGLE_MIN_DEG:g}-85 deg')
    if not 0 <= cfg.height_ft <= 20 or not 0 < cfg.cd <= 2:
        raise ValueError('Height 0-20 ft, Cd in (0, 2]')
    if min(cfg.wire_d, cfg.coil_d, cfg.n_active, cfg.free_len, cfg.tpi) <= 0 or cfg.n_total < cfg.n_active:
        raise ValueError('Spring geometry must be positive and total coils >= active coils')
    if cfg.coil_d <= cfg.wire_d:
        raise ValueError('Mean coil diameter must be larger than the wire')
    if not 0 <= cfg.p_min_in < cfg.p_max_in < cfg.x_cock_in:
        raise ValueError('Need 0 <= stopper preload min < max < latch compression')
    if cfg.x_cock_in >= cfg.free_len - solid_len(cfg):
        raise ValueError('Latch compression reaches solid height')
    if min(cfg.m_piston_assy, cfg.m_stopper, cfg.rod_len_in) < 0 or not 0 < cfg.eta <= 1:
        raise ValueError('Masses >= 0 and 0 < eta <= 1')
    if cfg.material not in MATERIALS or cfg.rod_material not in ROD_DENSITY:
        raise ValueError('Unknown material')


# ================================================================= tests ====
def self_test():
    base = Config()
    # spring from geometry reproduces the spec
    s = spring_check(base)
    assert abs(s['k'] - 12.5) < 0.05 and abs(s['mass'] - 0.78) < 0.02 and abs(s['fn'] - 39) < 1.5, s
    assert abs(m_hardware(base) - 1.037) < 0.005
    # current (as-built) design passes: worst-case ball reaches 20 ft, every check OK
    assert s['verdict'] == 'PASS' and all(s['chk'].values()) and 0 < s['margin'] < 0.20, (s['margin'], s['chk'])
    assert design_study(base)['B'] is None
    # heavier piston side -> FAIL; option A hits the target exactly, option B passes every check
    m_heavy = m_hardware(base) + 0.4
    A = fix_latch(base, m_heavy)
    assert abs(A['margin'] - base.margin_target) < 1e-6 and A['cfg'].x_cock_in > base.x_cock_in
    Bs, n, _ = fix_spring(base, m_heavy)
    assert Bs and all(Bs[0]['chk'].values()) and Bs[0]['verdict'] == 'PASS', n
    # max mass is consistent with the margin definition
    for tgt in (0.0, 0.1):
        assert abs(margin(base, max_hw_mass(base, tgt)) - tgt) < 1e-9
    # heavier piston side: slower, and the verdict degrades to FAIL
    st = design_study(base, m_whatif=m_hardware(base) + 0.4)
    assert st['v_cur'] < st['v_req'] and st['cur']['verdict'] == 'FAIL' and not st['reach']
    assert st['B'] is None or (all(st['B']['chk'].values()) and st['B']['cfg'].free_len <= base.l0_max + 1e-9)
    # tab 1 with a too-heavy design reports the miss instead of crashing
    heavy = replace(base, diameter_in=4.75, m_piston_assy=base.m_piston_assy + 0.4)
    sh = aim(heavy)
    assert not sh['ok'] and sh['landing'] < TARGET
    # aiming on the current design
    n_ok = 0
    for d in (1.9, 2.5, 3.0, 4.0, 4.5, 4.75):
        c = replace(base, diameter_in=d)
        so = aim(c)
        assert so['ok'] and abs(flight(c, so)['range'] - TARGET) < 0.01, (d, so)
        if so['mode'] == 'angle':
            assert so['p'] == so['p0'] and so['branch'] in ('low', 'peak')
        n_ok += 1
    narrow = replace(base, diameter_in=1.9, angle_min=36.0, angle_max=50.0)
    so = aim(narrow)
    assert so['mode'] == 'stopper' and so['angle'] == narrow.angle_min and abs(flight(narrow, so)['range'] - TARGET) < 0.01
    for d in (1.9, 3.0, 4.75):                     # never below the 31 deg hard limit, never the high branch
        so = aim(replace(base, diameter_in=d))
        assert so['angle'] >= ANGLE_MIN_DEG - 1e-6 and so['branch'] != 'high', (d, so['angle'])
    tight = replace(base, diameter_in=1.9, angle_min=35.0, angle_max=39.0)
    so = aim(tight)
    assert so['mode'] == 'stopper' and so['p'] > so['p0'] and abs(flight(tight, so)['range'] - TARGET) < 0.01
    for a in (30, 37, 45):
        for p in (0.6, 1.3, 2.5):
            assert abs(preload_for_speed(base, a, launch_speed(base, a, p)) - p) < 1e-9
    # COMSOL-check model equals launch_speed when masses are mapped the same way (eta = 1)
    cc = replace(base, eta=1.0)
    hw = 0.6
    v1 = comsol_check(spring_k(cc), cc.x_cock_in, cc.x_cock_in - cc.p_min_in, 0.5, spring_mass(cc) / 6, hw / 2, 0.0, 37)
    v2 = math.sqrt(2 * (spring_energy(cc, cc.p_min_in) - moving_mass(cc, hw) * math.sin(math.radians(37))
                        * (cc.x_cock_in - cc.p_min_in)) / 12 / (moving_mass(cc, hw) / LBM_PER_SLUG))
    assert abs(v1 - v2) < 1e-9, (v1, v2)
    print(f"PASS: spring k {s['k']:.2f} lbf/in, {s['mass']:.2f} lb, {s['fn']:.0f} Hz; current design "
          f"{s['verdict']} (margin {s['margin']*100:+.1f}%); +0.4 lb: A latch {A['cfg'].x_cock_in:.2f} in; B {Bs[0]['cfg'].wire_d} in wire "
          f"k {Bs[0]['k']:.2f} ({n} tried); {n_ok} balls on target; 31 deg floor + stopper paths; FAIL path")


# =============================================================== figures ====
BG = '#fbfcff'
GREEN, RED, AMBER, BLUE, INK, MUTED, TEAL = '#17774b', '#b44136', '#b8650f', '#1173b7', '#324f67', '#72879b', '#0e766e'
VCOL = {'PASS': GREEN, 'MARGINAL': AMBER, 'FAIL': RED}
CHK_LABEL = {'solid': 'solid SF', 'goodman': 'Goodman SF', 'clear': 'clearance to solid', 'buckle': 'buckling',
             'od': 'OD', 'id': 'ID', 'force': 'cocking force', 'margin': 'energy margin'}


def _style(ax):
    ax.set_facecolor('#f5f8fc'); ax.grid(True, color='#dbe5ef', lw=.7); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def _mode_line(sol):
    if not sol['ok']:
        return f"CANNOT REACH 20 FT: lands {(sol['landing']-TARGET)*12:+.1f} in - {sol['note']}"
    if sol['mode'] == 'angle':
        b = {'low': 'low-angle solution', 'peak': 'max-range angle', 'high': 'HIGH-angle solution'}[sol['branch']]
        return f"Aimed by ANGLE only ({b}); stopper stays at baseline {sol['turns0']:+.1f} turns"
    return f"Angle range not enough -> STOPPER {sol['turns0']:+.1f} -> {sol['turns']:+.1f} turns ({sol['note']})"


def make_tab1_figure(cfg: Config, sol, fl, table, dpi=120):
    import textwrap
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    fig = Figure(figsize=(13.2, 10.0), dpi=dpi, facecolor=BG)
    fig.text(.06, .955, 'BALL TRAJECTORY & AIMING  /  20 FT TARGET', fontsize=20, fontweight='bold', color='#10243a')
    mcol = RED if not sol['ok'] else (AMBER if sol['high_warning'] or sol['mode'] == 'stopper' else MUTED)
    fig.text(.061, .925, _mode_line(sol), fontsize=11, color=mcol, fontweight='bold' if not sol['ok'] else 'normal')

    ax = fig.add_axes([.07, .52, .86, .38]); _style(ax)
    ax.set_xlim(-1.5, 24); ax.set_ylim(-.5, max(9, fl['apex_y'] * 1.35))
    ax.axhline(0, color='#567489', lw=2)
    ax.add_patch(Rectangle((-0.6, 0), .8, max(.15, cfg.height_ft), facecolor='#7b92a7', edgecolor='#34495e', zorder=4))
    th = math.radians(sol['angle'])
    ax.plot([0, 1.4 * math.cos(th)], [cfg.height_ft, cfg.height_ft + 1.4 * math.sin(th)],
            color='#263c51', lw=9, solid_capstyle='round', zorder=6)
    col = RED if (sol['high_warning'] or not sol['ok']) else BLUE
    ax.plot(fl['x'], fl['y'], color=col, lw=3, zorder=7, label=f"Ø {cfg.diameter_in:.2f} in  →  {fl['range']:.2f} ft")
    n = len(fl['x'])
    idx = [round(q * (n - 1)) for q in (.0, .27, .54, .78, 1.0)]
    ax.scatter([fl['x'][i] for i in idx], [fl['y'][i] for i in idx], s=60 + 20 * cfg.diameter_in,
               color='#f6a740', edgecolors='#8d4c0b', zorder=12)
    ax.annotate(f"APEX {fl['apex_y']:.2f} ft", (fl['apex_x'], fl['apex_y']), xytext=(0, -22), ha='center',
                textcoords='offset points', fontsize=10, fontweight='bold', color=TEAL)
    ax.axvspan(TARGET - TOL_FT, TARGET + TOL_FT, color='#fde7d6', zorder=0, label='±6 in')
    ax.scatter([TARGET], [0], marker='X', s=180, color='#d94d42', zorder=16)
    ax.text(.02, .95, f"θ = {sol['angle']:.1f}°   v₀ = {sol['v']:.2f} ft/s   h = {cfg.height_ft:.2f} ft   "
            f"ball Ø {cfg.diameter_in:.2f} in   stopper {sol['turns']:+.1f} turns", transform=ax.transAxes,
            va='top', fontsize=11, color='#08705b', fontweight='bold',
            bbox=dict(boxstyle='round,pad=.4', facecolor='white', edgecolor='#a2d3c5'))
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('Horizontal position (ft)'); ax.set_ylabel('Height (ft)')

    y = .455
    if sol['high_warning']:
        for line in textwrap.wrap('⚠ ' + warning_text(sol), 165):
            fig.text(.07, y, line, fontsize=9.5, fontweight='bold', color=RED); y -= .022
    fig.text(.07, y, f"Flight {fl['t']:.3f} s  |  apex {fl['apex_y']:.2f} ft  |  ball KE {fl['ke_ball']:.0f} in-lb  |  "
             f"aim error {sol['dRdth_in']:+.1f} in/deg  |  {WIND_CHECK_MPH:g} mph headwind {sol['wind_in']:+.1f} in  |  "
             f"Cd ±10% {sol['cd_hi_in']:+.1f} / {sol['cd_lo_in']:+.1f} in", fontsize=9.5, color=INK)

    # bottom-left: plan vs ball size
    bx = fig.add_axes([.07, .06, .36, .30]); _style(bx)
    ds = [d for d, _ in table]
    bx.axhspan(cfg.angle_min, cfg.angle_max, color='#eef3f8', zorder=0)
    bx.plot(ds, [s['angle'] for _, s in table], color=BLUE, lw=2.5, marker='o', label='aim angle')
    for d, s in table:
        if s['high_warning'] or not s['ok']:
            bx.scatter([d], [s['angle']], s=140, facecolors='none', edgecolors=RED, lw=2, zorder=6)
    bx.axvline(cfg.diameter_in, color=MUTED, ls=':', lw=1)
    bx.set_xlabel('Ball diameter (in)'); bx.set_ylabel('Launch angle (deg)')
    bx.set_title('Ball-size plan (red ring = high angle / miss)', fontsize=10, fontweight='bold')
    tx = bx.twinx()
    tx.plot(ds, [s['turns'] for _, s in table], color=AMBER, lw=2, ls='--', marker='s', label='stopper turns')
    tx.set_ylabel('Stopper turns', color=AMBER)
    tlo, thi = min(s['turns'] for _, s in table), max(s['turns'] for _, s in table)
    tx.set_ylim(tlo - 3, thi + 3)
    h1, l1 = bx.get_legend_handles_labels(); h2, l2 = tx.get_legend_handles_labels()
    bx.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')

    # bottom-right: landing vs angle at the baseline stopper
    cx = fig.add_axes([.57, .06, .37, .30]); _style(cx)
    angs = [cfg.angle_min + (cfg.angle_max - cfg.angle_min) * i / 40 for i in range(41)]
    seen = []
    for d, c in ((cfg.diameter_in, BLUE), (DIAM_MIN_IN, TEAL), (DIAM_MAX_IN, AMBER)):
        if any(abs(d - q) < 1e-9 for q in seen):
            continue
        seen.append(d)
        cc = replace(cfg, diameter_in=d)
        cx.plot(angs, [landing_with_stopper(cc, a, sol['p0']) for a in angs], color=c, lw=2.6 if c == BLUE else 1.8,
                label=f'Ø {d:g} in' + ('  (this ball)' if c == BLUE else ''))
    cx.axhspan(TARGET - TOL_FT, TARGET + TOL_FT, color='#fde7d6', zorder=0)
    cx.axhline(TARGET, color='#d94d42', lw=1, ls=':')
    cx.axvline(cfg.angle_nominal, color=MUTED, ls=':', lw=1)
    cx.set_ylim(TARGET - 3, TARGET + 1.2)
    cx.set_xlabel('Launch angle (deg)'); cx.set_ylabel('Landing (ft)')
    cx.set_title(f"Landing vs angle at baseline stopper {sol['turns0']:+.1f} turns", fontsize=10, fontweight='bold')
    cx.legend(fontsize=8, loc='lower center')
    return fig


def _fmt_spring_rows(st):
    """Rows for the simple Current vs B table: (label, [cells], [fail flags])."""
    alt = st['B'] or st.get('B_close')
    cols = [st['cur']] + ([alt] if alt else [])
    rows = []

    def add(label, fn, bad=None):
        cells, flags = [], []
        for s in cols:
            cells.append('—' if s is None else fn(s))
            flags.append(False if s is None or bad is None else bad(s))
        rows.append((label, cells, flags))

    add('Wire d / OD  (in)', lambda s: f"{s['cfg'].wire_d:.3f} / {s['OD']:.2f}",
        lambda s: not (s['chk']['od'] and s['chk']['id']))
    add('Active / total coils', lambda s: f"{s['cfg'].n_active:.1f} / {s['cfg'].n_total:.1f}")
    add('Free length  (in)', lambda s: f"{s['cfg'].free_len:.2f}  (= {s['Ls']:.2f} + 1.15 × {s['cfg'].x_cock_in:g})"
        if abs(s['cfg'].free_len - (s['Ls'] + 1.15 * s['cfg'].x_cock_in)) < 0.06 else f"{s['cfg'].free_len:.2f}",
        lambda s: not s['chk']['clear'])
    add('Rate k  (lbf/in)', lambda s: f"{s['k']:.2f}")
    add('Cocking force  (lbf)', lambda s: f"{s['F_latch']:.0f}", lambda s: not s['chk']['force'])
    add('Energy margin', lambda s: f"{s['margin']*100:+.1f}%  {s['verdict']}", lambda s: s['verdict'] != 'PASS')
    add(f'Goodman SF  (≥{SF_GOODMAN_MIN})', lambda s: f"{s['n_goodman']:.2f}", lambda s: not s['chk']['goodman'])
    add(f'Solid SF  (≥{SF_SOLID_MIN})', lambda s: f"{s['sf_solid']:.2f}", lambda s: not s['chk']['solid'])
    add('Spring mass  (lb)', lambda s: f"{s['mass']:.2f}")
    add('All checks', lambda s: 'OK' if all(s['chk'].values()) else
        'FAILS: ' + ', '.join(CHK_LABEL[k] for k, v in s['chk'].items() if not v),
        lambda s: not all(s['chk'].values()))
    return rows


def make_tab2_figure(cfg: Config, st, dpi=120):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(13.2, 9.0), dpi=dpi, facecolor=BG)
    cur, B = st['cur'], st['B']
    fig.text(.05, .955, 'LAUNCHER DESIGN  /  IS THE SPRING ENOUGH?', fontsize=20, fontweight='bold', color='#10243a')
    if cur['verdict'] == 'PASS':
        head, col = f"Current spring: PASS (margin {cur['margin']*100:+.1f}%)  -  no change needed", GREEN
    elif B:
        head = (f"Current spring: {cur['verdict']} (margin {cur['margin']*100:+.1f}%)   →   "
                f"Recommended: new spring B (margin {B['margin']*100:+.1f}%, PASS)")
        col = VCOL[cur['verdict']]
    else:
        head, col = f"Current spring: {cur['verdict']} (margin {cur['margin']*100:+.1f}%)  -  no new spring found", RED
    fig.text(.051, .915, head, fontsize=13, fontweight='bold', color=col)
    who = 'what-if' if st['whatif'] else 'design'
    fig.text(.051, .885, f"Worst case Ø{cfg.baseline_ball_in:g} in ball @ {cfg.angle_nominal:g}° needs v₀ {st['v_req']*0.3048:.2f} m/s  |  "
             f"full stroke gives {launch_speed(cfg, cfg.angle_nominal, cfg.p_min_in, st['m_an'])*0.3048:.2f} m/s  |  "
             f"piston-side hardware {st['m_an']:.2f} lb ({who})  |  pass = worst-case ball reaches 20 ft"
             + (f" with {cfg.margin_target*100:.0f}% margin" if cfg.margin_target > 0 else ""),
             fontsize=10, color=MUTED)

    # one graph: landing of the worst-case ball vs piston-side mass, stopper fully open
    sw = st['sweep']
    ax = fig.add_axes([.07, .47, .86, .37]); _style(ax)
    ax.axhspan(TARGET - TOL_FT, TARGET + TOL_FT, color='#fde7d6', zorder=0)
    ax.axhline(TARGET, color='#d94d42', lw=1.2, ls=':', label='20 ft target')
    ax.plot(sw['m'], sw['land_cur'], color=BLUE, lw=2.6, label='current spring')
    if sw['land_B']:
        ax.plot(sw['m'], sw['land_B'], color=GREEN, lw=2.6, ls='--', label='new spring B')
    ax.axvline(st['m_design'], color=INK, lw=1, ls=':')
    ax.scatter([st['m_design']], [landing_at(replace(cfg, diameter_in=cfg.baseline_ball_in), cfg.angle_nominal,
                launch_speed(cfg, cfg.angle_nominal, cfg.p_min_in, st['m_design']))],
               s=60, color=INK, zorder=6, label=f"design {st['m_design']:.2f} lb")
    if st['whatif']:
        ax.scatter([st['m_an']], [st['land_full']], s=90, marker='D', color=AMBER, zorder=7,
                   label=f"what-if {st['m_an']:.2f} lb → {st['land_full']:.2f} ft")
    lo = min(min(sw['land_cur']), TARGET - 1.5)
    hi = max(max(sw['land_B'] or sw['land_cur']), TARGET + 1)
    ax.set_ylim(max(lo, TARGET - 4), min(hi, TARGET + 4))
    lims = [(st['m_max_zero'], BLUE, 'current')] + ([(st['m_max_zero_B'], GREEN, 'B')] if B else [])
    y0 = ax.get_ylim()[0]
    for j, (mlim, c, name) in enumerate(lims):       # mass where the curve crosses 20 ft
        if sw['m'][0] < mlim < sw['m'][-1]:
            ax.axvline(mlim, color=c, lw=1, alpha=.6)
            ax.text(mlim, y0 + 0.15 + 0.55 * j, f" {name}: max {mlim:.2f} lb", fontsize=9, color=c,
                    fontweight='bold', va='bottom')
    ax.set_xlabel('Piston-side hardware mass, without spring (lb)')
    ax.set_ylabel(f'Landing of Ø{cfg.baseline_ball_in:g} in ball (ft)')
    ax.set_title('Does the worst-case ball still reach 20 ft?  (stopper fully open = max energy)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    # small table: current vs B
    tx = fig.add_axes([.12, .07, .76, .32]); tx.axis('off')
    rows = _fmt_spring_rows(st)
    heads = ['', 'CURRENT spring']
    if B or st.get('B_close'):
        heads.append('NEW SPRING B' if B else 'B: closest (none passes)')
    widths = [.30, .35, .35] if len(heads) == 3 else [.40, .45]
    tbl = tx.table(cellText=[[r[0]] + r[1] for r in rows], colLabels=heads,
                   loc='upper center', cellLoc='center', colWidths=widths)
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.32)
    for (r, c), cellobj in tbl.get_celld().items():
        cellobj.set_edgecolor('#dbe5ef')
        if r == 0:
            cellobj.set_facecolor('#10243a'); cellobj.get_text().set_color('white'); cellobj.get_text().set_fontweight('bold')
        elif c == 0:
            cellobj.get_text().set_ha('left'); cellobj.set_facecolor('#f5f8fc')
        else:
            bad = rows[r - 1][2][c - 1]
            cellobj.set_facecolor('#fbe3e1' if bad else 'white')
            if rows[r - 1][0] in ('Energy margin', 'All checks'):
                cellobj.get_text().set_fontweight('bold'); cellobj.get_text().set_color(RED if bad else GREEN)
    if st['A']:
        a = st['A']
        ok = all(a['chk'].values())
        fig.text(.12, .065, f"Moving the latch alone ({cfg.x_cock_in:g} → {a['cfg'].x_cock_in:.2f} in, same spring) "
                 + ('also works.' if ok else f"is not enough: clearance to solid {a['clear_frac']*100:.0f}% < "
                    f"{cfg.solid_clear*100:.0f}%."), fontsize=9.5, color=AMBER)
    return fig


def table_text(cfg, table):
    lines = [' Ø in  angle   turns  stroke  mode']
    for d, s in table:
        tag = s['mode'] + (' HIGH!' if s['high_warning'] else '') + ('' if s['ok'] else ' SHORT')
        lines.append(f"{d:4.2f} {s['angle']:6.1f} {s['turns']:+7.1f} {s['stroke']:7.3f}  {tag}")
    return '\n'.join(lines)


def write_csv(cfg, table, path):
    with open(path, 'w') as fh:
        fh.write('ball_diameter_in,angle_deg,stopper_preload_in,stopper_turns,stroke_in,v0_ftps,landing_ft,'
                 'mode,branch,high_angle_warning,reaches_20ft\n')
        for d, s in table:
            fh.write(f"{d:.2f},{s['angle']:.2f},{s['p']:.4f},{s['turns']:.2f},{s['stroke']:.4f},{s['v']:.3f},"
                     f"{s['landing']:.3f},{s['mode']},{s['branch']},{s['high_warning']},{s['ok']}\n")


def design_text(cfg, st):
    cur, B = st['cur'], st['B']
    v_full = launch_speed(cfg, cfg.angle_nominal, cfg.p_min_in, st['m_an'])
    t0 = (f"v0 full stroke   {v_full*0.3048:5.2f} m/s\n"
          f"  (stopper open, {cfg.angle_nominal:g} deg)\n"
          f"v0 needed        {st['v_req']*0.3048:5.2f} m/s\n\n")
    t = t0 + (f"CURRENT  {cur['verdict']}  ({cur['margin']*100:+.1f}%)\n"
         f"  k {cur['k']:.2f} lbf/in, {cur['F_latch']:.0f} lbf\n"
         f"  max hardware {st['m_max_zero']:.2f} lb\n"
         f"Hardware now {st['m_an']:.2f} lb{' (what-if)' if st['whatif'] else ''}\n"
         f"  worst ball lands {st['land_full']:.2f} ft\n"
         f"  (stopper fully open)")
    if B:
        b = B['cfg']
        t += (f"\n\nNEW SPRING B  PASS ({B['margin']*100:+.1f}%)\n"
              f"  d {b.wire_d:.3f}, D {b.coil_d:.2f}, Na {b.n_active:.1f}\n"
              f"  L0 {b.free_len:.2f} in, k {B['k']:.2f}\n"
              f"  {B['F_latch']:.0f} lbf, {B['mass']:.2f} lb\n"
              f"  max hardware {st['m_max_zero_B']:.2f} lb")
    elif cur['verdict'] != 'PASS':
        t += f"\n\nNo new spring passes ({st['b_tried']} tried)"
    return t


# =================================================================== GUI ====
def main():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title('Ball Launcher | Aiming & Design')
    root.geometry('1500x920'); root.minsize(900, 560)
    style = ttk.Style(root)
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    style.configure('TButton', padding=(10, 7))
    # fixed tab size: clam enlarges the selected tab by default ("expand"), switch that off
    style.configure('TNotebook.Tab', padding=(18, 8), font=('Segoe UI', 11, 'bold'))
    style.map('TNotebook.Tab', padding=[('selected', (18, 8))], expand=[('selected', (0, 0, 0, 0))])

    nb = ttk.Notebook(root); nb.pack(fill='both', expand=True, padx=10, pady=10)
    tab1, tab2 = ttk.Frame(nb, padding=10), ttk.Frame(nb, padding=10)
    nb.add(tab1, text='1. Trajectory & Aiming'); nb.add(tab2, text='2. Launcher Design')

    def scroll_panel(parent, width=370):
        outer = ttk.Frame(parent, width=width); outer.pack(side='left', fill='y', padx=(0, 12)); outer.pack_propagate(False)
        cv = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set); sb.pack(side='right', fill='y'); cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv, padding=(0, 0, 7, 10)); w = cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        cv.bind('<Configure>', lambda e: cv.itemconfigure(w, width=e.width))

        def wheel(e):
            n = -1 if e.num == 4 else 1 if e.num == 5 else (-int(e.delta / 120) if abs(e.delta) >= 120 else (-1 if e.delta > 0 else 1))
            cv.yview_scroll(n, 'units'); return 'break'
        outer.bind('<Enter>', lambda e: [cv.bind_all(s, wheel) for s in ('<MouseWheel>', '<Button-4>', '<Button-5>')])
        outer.bind('<Leave>', lambda e: [cv.unbind_all(s) for s in ('<MouseWheel>', '<Button-4>', '<Button-5>')])
        return inner

    fields, combos = {}, {}

    def section(parent, title):
        fr = ttk.LabelFrame(parent, text=title, padding=10); fr.pack(fill='x', pady=(0, 10)); return fr

    def add(frame, label, key, val):
        ttk.Label(frame, text=label).pack(anchor='w', pady=(5, 2))
        v = tk.StringVar(value=val)
        ent = ttk.Entry(frame, textvariable=v); ent.pack(fill='x', ipady=2); fields[key] = v
        return ent

    def add_combo(frame, label, key, values, val):
        ttk.Label(frame, text=label).pack(anchor='w', pady=(5, 2))
        v = tk.StringVar(value=val)
        ttk.Combobox(frame, textvariable=v, values=list(values), state='readonly').pack(fill='x')
        combos[key] = v

    # ---- tab 1
    L1 = scroll_panel(tab1)
    s = section(L1, 'BALL')
    add(s, f'Ball diameter (in)  [{DIAM_MIN_IN}–{DIAM_MAX_IN}]', 'diam', f'{DIAM_DEFAULT_IN:g}')
    ttk.Label(s, text=f'{MASS_BALL_LBM:g} lb, target {TARGET:.0f} ft, drag always on', foreground='#bf453c').pack(anchor='w', pady=(6, 0))
    s = section(L1, 'SETUP')
    fields['height'] = tk.StringVar(value=f'{HEIGHT_FT:g}')      # fixed, no input box
    ttk.Label(s, text=f'Release height: {HEIGHT_FT:g} ft', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(5, 2))
    add(s, 'Drag coefficient Cd', 'cd', f'{CD:g}')
    fields['amin'] = tk.StringVar(value=f'{ANGLE_MIN_DEG:g}')    # fixed hard limit, no input box
    ttk.Label(s, text=f'Minimum angle: {ANGLE_MIN_DEG:g} deg (hard limit, ball rolls off below this)',
              font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(5, 2))
    add(s, 'Angle range max (deg)', 'amax', f'{ANGLE_MAX_DEG:g}')
    s = section(L1, 'STOPPER')
    manual_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(s, text='Manual stopper turns (else auto baseline)', variable=manual_var).pack(anchor='w')
    add(s, 'Manual stopper turns', 'turns', '35')
    ttk.Label(s, justify='left', wraplength=320, foreground='#324f67', text=(
        f'1 turn = {1/THREAD_TPI:.3f} in of stopper travel ({THREAD_TPI:g} TPI, 1/2-20 UNF).\n'
        f'Turns are counted from the fully open stopper (preload {P_MIN_IN:g} in):\n'
        f'  stroke = {X_COCK_IN - P_MIN_IN:g} in - {1/THREAD_TPI:.3f} in x turns\n'
        f'  0 turns -> stroke {X_COCK_IN - P_MIN_IN:.2f} in (max energy, COMSOL case)\n'
        f'  35 turns -> stroke {X_COCK_IN - P_MIN_IN - 35/THREAD_TPI:.2f} in (about 6.95 m/s, 20 ft setting)\n'
        f'Each turn changes the stored energy by about k x preload x {1/THREAD_TPI:.3f} in\n'
        f'(about 1.4 in-lbf, 0.6 %, at the 20 ft setting).')).pack(anchor='w', pady=(6, 0))
    out1, warn1, tab_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
    r1 = section(L1, 'AIM')
    ttk.Label(r1, textvariable=out1, font=('Consolas', 10), justify='left').pack(anchor='w')
    ttk.Label(r1, textvariable=warn1, foreground='#b44136', wraplength=320, justify='left',
              font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(8, 0))
    ttk.Label(section(L1, 'BALL-SIZE PLAN'), textvariable=tab_var, font=('Consolas', 9), justify='left').pack(anchor='w')
    R1 = ttk.Frame(tab1); R1.pack(side='left', fill='both', expand=True)

    # ---- tab 2 (simple view; advanced inputs folded away)
    L2 = scroll_panel(tab2)
    s = section(L2, 'SPRING (current)')
    add(s, 'Wire diameter d (in)', 'wire', f'{WIRE_D:g}')
    add(s, 'Mean coil diameter D (in)', 'coil', f'{COIL_D:g}')
    add(s, 'Active coils', 'na', f'{N_ACTIVE:g}')
    add(s, 'Total coils', 'nt', f'{N_TOTAL:g}')
    add(s, 'Free length (in)', 'L0', f'{FREE_LEN:g}')
    add(s, 'Compression at latch x_L (in)', 'xc', f'{X_COCK_IN:g}')
    s = section(L2, 'PISTON SIDE  (spring/3 added automatically)')
    add(s, 'Piston + nut + washer + loop (lb)', 'mpist', f'{M_PISTON_ASSY:g}')
    add(s, 'Stopper + jam nut + washers (lb)', 'mstop', f'{M_STOPPER:g}')
    add(s, f'Tail rod length (in), Ø{ROD_D_IN:g}', 'rodlen', f'{ROD_LEN_IN:g}')
    add_combo(s, 'Tail rod material', 'rodmat', ROD_DENSITY.keys(), ROD_MATERIAL)
    s = section(L2, 'COMSOL CHECK  (same inputs as the COMSOL half model, angle 0, no friction)')
    add(s, 'Stroke (in)  [6 = stopper fully open]', 'c_stroke', '6')
    add(s, 'Ball mass, half model (lb)', 'c_ball', '0.5')
    add(s, 'Spring effective mass, half model (lb)', 'c_spr', '0.13')
    add(s, 'Added mass rod+stopper+hardware, half (lb)', 'c_add', '0.32')
    add(s, 'Piston mass, half model (lb)  [COMSOL domain mass]', 'c_pis', '0.255')
    add(s, 'COMSOL release speed (m/s)', 'c_res', '7.35')
    out3 = tk.StringVar()
    ttk.Label(s, textvariable=out3, font=('Consolas', 10, 'bold'), justify='left').pack(anchor='w', pady=(6, 0))
    s = section(L2, 'WHAT-IF')
    add(s, 'Piston-side hardware mass (lb), blank = design', 'whatif', '')
    out2 = tk.StringVar()
    ttk.Label(section(L2, 'RESULT'), textvariable=out2, font=('Consolas', 9), justify='left').pack(anchor='w')
    actions2 = ttk.Frame(L2); actions2.pack(fill='x')
    adv_holder = ttk.Frame(L2); adv_holder.pack(fill='x', pady=(10, 0))
    adv = ttk.Frame(adv_holder)
    adv_btn = ttk.Button(adv_holder, text='▸ Advanced settings')
    adv_btn.pack(fill='x')

    def toggle_adv():
        if adv.winfo_ismapped():
            adv.pack_forget(); adv_btn.configure(text='▸ Advanced settings')
        else:
            adv.pack(fill='x', pady=(6, 0)); adv_btn.configure(text='▾ Advanced settings')
    adv_btn.configure(command=toggle_adv)
    s = section(adv, 'MATERIAL & LOSSES')
    add_combo(s, 'Spring material', 'material', MATERIALS.keys(), MATERIAL)
    peen_var = tk.BooleanVar(value=PEENED)
    ttk.Checkbutton(s, text='Shot-peened', variable=peen_var).pack(anchor='w', pady=(6, 0))
    add(s, 'Efficiency η (calibrate)', 'eta', f'{ETA:g}')
    s = section(adv, 'STOPPER')
    add(s, 'Preload min (in)', 'pmin', f'{P_MIN_IN:g}')
    add(s, 'Preload max (in)', 'pmax', f'{P_MAX_IN:g}')
    add(s, 'Thread (TPI)', 'tpi', f'{THREAD_TPI:g}')
    add(s, 'Design point: worst-case ball Ø (in)', 'dref', f'{DIAM_MAX_IN:g}')
    add(s, 'Design point: nominal angle (deg)', 'anom', f'{ANGLE_NOMINAL_DEG:g}')
    s = section(adv, 'PASS CRITERIA')
    add(s, 'Energy margin target (%)', 'mtarget', f'{MARGIN_TARGET*100:g}')
    add(s, 'Min clearance to solid (% of x_L)', 'clr', f'{SOLID_CLEAR_FRAC*100:g}')
    add(s, 'Max spring OD (in)', 'odmax', f'{OD_MAX:g}')
    add(s, 'Min spring ID (in)', 'idmin', f'{ID_MIN:g}')
    add(s, 'Max cocking force (lbf)', 'fmax', f'{F_COCK_MAX:g}')
    add(s, 'Max free length for a new spring (in)', 'l0max', f'{L0_MAX_IN:g}')
    R2 = ttk.Frame(tab2); R2.pack(side='left', fill='both', expand=True)

    state = {'c1': None, 'c2': None, 'f1': None, 'f2': None, 'cfg': None, 'table': None, 'key2': None,
             'st': None, 'was_high': False}

    def read_inputs():
        try:
            g = lambda k: float(fields[k].get())
            c = Config(diameter_in=g('diam'), height_ft=g('height'), cd=g('cd'), angle_min=g('amin'),
                       angle_max=g('amax'), angle_nominal=g('anom'), material=combos['material'].get(),
                       peened=peen_var.get(), wire_d=g('wire'), coil_d=g('coil'), n_active=g('na'),
                       n_total=g('nt'), free_len=g('L0'), x_cock_in=g('xc'), p_min_in=g('pmin'),
                       p_max_in=g('pmax'), tpi=g('tpi'), m_piston_assy=g('mpist'), m_stopper=g('mstop'),
                       rod_len_in=g('rodlen'), rod_material=combos['rodmat'].get(), eta=g('eta'),
                       baseline_ball_in=g('dref'), stopper_turns=g('turns') if manual_var.get() else None,
                       margin_target=g('mtarget') / 100, solid_clear=g('clr') / 100, od_max=g('odmax'),
                       id_min=g('idmin'), f_cock_max=g('fmax'), l0_max=g('l0max'))
            w = fields['whatif'].get().strip()
            m_w = float(w) if w else None
        except ValueError as e:
            raise ValueError('All inputs must be numbers.') from e
        validate(c)
        return c, m_w

    def show(fig, frame, key):
        if state[key] is not None:
            state[key].get_tk_widget().destroy()
        c = FigureCanvasTkAgg(fig, master=frame); c.draw(); c.get_tk_widget().pack(fill='both', expand=True)
        state[key] = c

    def update(_e=None):
        try:
            cfg, m_w = read_inputs()
            sol = aim(cfg); fl = flight(cfg, sol); table = ball_table(cfg)
            fig1 = make_tab1_figure(cfg, sol, fl, table, dpi=96)
            key2 = (design_key(cfg), m_w)
            st = state['st'] if key2 == state['key2'] else design_study(cfg, m_w)
            fig2 = None if key2 == state['key2'] else make_tab2_figure(cfg, st, dpi=96)
        except Exception as exc:
            messagebox.showerror('Input or calculation error', str(exc)); return
        show(fig1, R1, 'c1')
        if fig2 is not None:
            show(fig2, R2, 'c2'); state.update(f2=fig2, key2=key2, st=st)
        state.update(f1=fig1, cfg=cfg, table=table)
        how = 'ANGLE' if sol['mode'] == 'angle' else 'STOPPER'
        out1.set(f"Adjusted by     {how}{'' if sol['ok'] else '  (MISS)'}\n"
                 f"Launch angle    {sol['angle']:7.2f} deg ({sol['branch']})\n"
                 f"Stopper         {sol['turns']:+7.1f} turns\n"
                 f"  baseline      {sol['turns0']:+7.1f} turns\n"
                 f"Stroke          {sol['stroke']:7.3f} in\n"
                 f"Release v0      {sol['v']:7.2f} ft/s\n"
                 f"Landing         {fl['range']:7.3f} ft\n"
                 f"Flight time     {fl['t']:7.3f} s\n"
                 f"Aim error       {sol['dRdth_in']:+7.1f} in/deg\n"
                 f"Wind {WIND_CHECK_MPH:g} mph      {sol['wind_in']:+7.1f} in")
        w = warning_text(sol)
        if sol['mode'] == 'stopper' or not sol['ok']:
            w = (w + '\n' if w else '') + sol['note'] + '.'
        warn1.set(w)
        tab_var.set(table_text(cfg, table))
        out2.set(design_text(cfg, st))
        try:
            g = lambda k_: float(fields[k_].get())
            vc = comsol_check(spring_k(cfg), cfg.x_cock_in, g('c_stroke'), g('c_ball'), g('c_spr'), g('c_add'),
                              g('c_pis'), 0.0) * 0.3048   # COMSOL model: no barrel angle
            ref = g('c_res')
            diff = (vc / ref - 1) * 100 if ref > 0 else float('nan')
            out3.set(f"Python   {vc:6.3f} m/s\nCOMSOL   {ref:6.3f} m/s\nDiff     {diff:+6.2f} %  "
                     f"{'MATCH (<1%)' if abs(diff) < 1 else 'CHECK INPUTS'}")
        except ValueError:
            out3.set('Enter numbers in the COMSOL CHECK fields')
        if sol['high_warning'] and not state['was_high']:
            messagebox.showwarning('High-angle solution', warning_text(sol))
        state['was_high'] = sol['high_warning']

    def apply_option(which):
        st = state['st']
        if not st or not st.get(which):
            messagebox.showinfo('Nothing to apply', 'No new spring is needed (or none was found) for this design.'); return
        c = st[which]['cfg']
        if which == 'A':
            fields['xc'].set(f'{c.x_cock_in:.3f}')
        else:
            for k, v in (('wire', c.wire_d), ('coil', c.coil_d), ('na', c.n_active), ('nt', c.n_total), ('L0', c.free_len)):
                fields[k].set(f'{v:g}')
        update()

    def export(ext):
        tab = nb.index(nb.select())
        fig = state['f1'] if tab == 0 else state['f2']
        if fig is None:
            return
        name = ('aiming' if tab == 0 else 'launcher_design') if ext != 'csv' else 'ball_size_plan'
        types = {'png': [('PNG', '*.png')], 'svg': [('SVG', '*.svg')], 'csv': [('CSV', '*.csv')]}[ext]
        path = filedialog.asksaveasfilename(defaultextension='.' + ext, filetypes=types, initialfile=f'{name}.{ext}')
        if not path:
            return
        if ext == 'csv':
            write_csv(state['cfg'], state['table'], path)
        else:
            fig.savefig(path, dpi=220, facecolor=BG)
        messagebox.showinfo('Export complete', f'Saved: {path}')

    for panel, exts in ((L1, ('png', 'svg', 'csv')), (actions2, ('png', 'svg'))):
        ttk.Button(panel, text='CALCULATE  /  REDRAW', command=update).pack(fill='x', pady=(0, 5))
        if panel is actions2:
            ttk.Button(panel, text='USE NEW SPRING B', command=lambda: apply_option('B')).pack(fill='x', pady=(0, 5))
        row = ttk.Frame(panel); row.pack(fill='x')
        for e in exts:
            ttk.Button(row, text=f'EXPORT {e.upper()}', command=lambda e=e: export(e)).pack(side='left', fill='x', expand=True, padx=2)
    root.bind('<Return>', update)
    update()
    root.mainloop()


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
    elif '--demo' in sys.argv:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        cfg = Config()
        sol = aim(cfg); fl = flight(cfg, sol); table = ball_table(cfg)
        st = design_study(cfg)
        for fig, name in ((make_tab1_figure(cfg, sol, fl, table), 'tab1_aiming.png'),
                          (make_tab2_figure(cfg, st), 'tab2_design.png'),
                          (make_tab2_figure(cfg, design_study(cfg, m_hardware(cfg) + 0.4)), 'tab2_design_whatif_heavy.png')):
            FigureCanvasAgg(fig); fig.savefig(name, dpi=130, facecolor=BG)
        write_csv(cfg, table, 'ball_size_plan.csv')
        print(table_text(cfg, table)); print(); print(design_text(cfg, st))
    else:
        main()