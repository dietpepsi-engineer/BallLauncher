# 1 lb Ball Launcher Design Tool

Spring-piston launcher, 1 lb ball to 20 ft. Single Python file (`ball_launcher_2tab.py`).
It reproduces the trajectory, spring, mass and aiming numbers shown in the slides.

## How to run

Requires Python 3.8+ and matplotlib (Tkinter is included with standard Python).

```
pip install matplotlib
python ball_launcher_2tab.py               # GUI (two tabs)
python ball_launcher_2tab.py --self-test   # consistency checks, prints PASS
python ball_launcher_2tab.py --demo        # writes tab PNGs + ball-size plan CSV, no GUI
```

## What each tab does

**Tab 1: Trajectory & Aiming** (slides 6–9)
- Finds the release speed and angle that land the ball exactly at 20 ft (release height 5 ft, air drag always on, RK4 integration).
- Aiming rule: change the angle first (low-angle solution only, never below 31°). If 31° still lands long, stay at 31° and turn the stopper in.
- Ball-size plan: angle and stopper turns for Ø1.9–4.75 in balls.

**Tab 2: Launcher Design** (slides 11–15, 23)
- Spring rate and checks from the spring geometry (Shigley: Bergsträsser factor, static allowable, Zimmerli + Goodman fatigue, buckling, natural frequency).
- Energy balance from latch to stop, energy margin verdict (PASS / MARGINAL / FAIL), and the maximum piston-side mass the spring can carry.
- What-if mass: shows the effect of a mass change without changing tab 1.
- **COMSOL CHECK**: enter the COMSOL half-model inputs; the tool computes the release speed under the same conditions (no friction, no barrel angle) and reports the difference. Default inputs give 7.36 m/s vs COMSOL 7.35 m/s.

## Launcher model

```
1/2 k (x_L^2 - x_P^2) = (1/eta) [ 1/2 m_eff v0^2 + m_lift g s sin(theta) ],   s = x_L - x_P
m_eff  = m_move + m_spring/3     (kinetic energy)
m_lift = m_move + m_spring/2     (spring centre of mass rises s/2)
```

## Default values (match the slides)

| Item | Value |
|---|---|
| Release height / nominal angle / minimum angle | 5 ft / 37° / 31° (hard limit) |
| Ball size range | Ø1.9–4.75 in, 1 lb |
| Drag coefficient | C_d = 0.47 (smooth sphere) |
| Spring | Cr-Si A401 shot-peened, d 0.187 in, D 2.24 in, N_a/N_t 12.2/14.2, L_0 10.1 in → k ≈ 12.5 lbf/in |
| Latch compression / preload range | x_L = 6.5 in / x_P = 0.5–2.7 in (1/2-20 stopper, 0.05 in per turn) |
| Piston assembly / tail rod / stopper | 0.547 / 0.34 (7075-T6, Ø0.5 × 18 in, CAD mass) / 0.15 lb |
| Mechanical efficiency | η = 0.985 |

## Assumptions to confirm by test

- **η = 0.985** (≈ 1.5% loss from wear band and bushing friction). The worst-case ball still reaches 20 ft down to η ≈ 0.88.
- **C_d = 0.47** for all ball sizes.
- Bumper stiffness and damping (k_b ≈ 1,650 lbf/in, c_b ≈ 130 N·s/m) are estimates used only in the COMSOL model, not in this tool.
