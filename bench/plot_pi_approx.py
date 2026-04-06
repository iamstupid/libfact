"""
Compare pi(n) approximation functions against primecount ground truth.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi  # for Li(x)

# Ground truth from primecount
data = """100,25
316,65
1000,168
3160,446
10000,1229
31600,3399
100000,9592
316000,27273
1000000,78498
3160000,227512
10000000,664579
31600000,1950638
100000000,5761455
316000000,17071133
1000000000,50847534
3160000000,151772733
10000000000,455052511
31600000000,1366257814
100000000000,4118054813
316000000000,12423276776
1000000000000,37607912018
3160000000000,113904403412
10000000000000,346065536839
31600000000000,1051637450787
100000000000000,3204941750802
316000000000000,9767043352554
1000000000000000,29844570422669
3160000000000000,91174971464705
10000000000000000,279238341033925
31600000000000000,854903053736865
100000000000000000,2623557157654233
316000000000000000,8047342268055800
1000000000000000000,24739954287740860"""

ns, pis = [], []
for line in data.strip().split('\n'):
    n, pi = line.split(',')
    ns.append(int(n))
    pis.append(int(pi))

ns = np.array(ns, dtype=np.float64)
pis = np.array(pis, dtype=np.float64)

# --- Approximation functions ---

def pi_xlogx(x):
    """n / ln(n) — simplest"""
    return x / np.log(x)

def pi_xlogx2(x):
    """n / (ln(n) - 1) — better classical"""
    return x / (np.log(x) - 1)

def pi_xlogx3(x):
    """n / (ln(n) - 1 - 1/ln(n)) — next term"""
    lx = np.log(x)
    return x / (lx - 1 - 1/lx)

def pi_li(x):
    """Li(x) = integral from 2 to x of 1/ln(t) dt = Ei(ln(x)) - Ei(ln(2))"""
    return np.array([expi(np.log(xi)) - expi(np.log(2)) for xi in x])

def pi_li_corrected(x):
    """Li(x) - Li(x^0.5)/2 — Riemann's correction (first term)"""
    li_x = np.array([expi(np.log(xi)) - expi(np.log(2)) for xi in x])
    li_sqrt = np.array([expi(np.log(xi**0.5)) - expi(np.log(2)) for xi in x])
    return li_x - li_sqrt / 2

def pi_rosser_upper(x):
    """Rosser-Schoenfeld upper bound: x/(ln(x)-4) for x >= 55"""
    return x / (np.log(x) - 4)

def pi_rosser_lower(x):
    """Rosser-Schoenfeld-style lower: x/(ln(x)) (always underestimates)"""
    return x / np.log(x)

# --- Compute approximations ---
approxs = {
    'n/ln(n)':               pi_xlogx(ns),
    'n/(ln(n)-1)':           pi_xlogx2(ns),
    'n/(ln(n)-1-1/ln(n))':   pi_xlogx3(ns),
    'Li(x)':                 pi_li(ns),
    'Li(x)-Li(√x)/2':       pi_li_corrected(ns),
}

# --- Plot 1: Absolute values (log-log) ---
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

ax1 = axes[0]
ax1.loglog(ns, pis, 'ko-', markersize=4, linewidth=2, label='π(n) exact', zorder=5)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for (name, vals), color in zip(approxs.items(), colors):
    ax1.loglog(ns, vals, '--', color=color, linewidth=1.5, label=name)
ax1.set_xlabel('n')
ax1.set_ylabel('π(n)')
ax1.set_title('Prime counting function: exact vs approximations')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Plot 2: Relative error ---
ax2 = axes[1]
for (name, vals), color in zip(approxs.items(), colors):
    rel_err = (vals - pis) / pis * 100
    ax2.semilogx(ns, rel_err, 'o-', color=color, markersize=3, linewidth=1.5, label=name)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xlabel('n')
ax2.set_ylabel('Relative error (%)')
ax2.set_title('Relative error of π(n) approximations')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('c:/Users/zball/Documents/bigint/libfact/bench/pi_approx.png', dpi=150)
# plt.show()
print("Saved to bench/pi_approx.png")

# --- Print table ---
print(f"\n{'n':>20s} {'π(n)':>20s}", end='')
for name in approxs:
    print(f" {name:>20s}", end='')
print(f" {'best_err%':>10s}")

for i in range(len(ns)):
    n, pi = ns[i], pis[i]
    print(f"{n:20.0f} {pi:20.0f}", end='')
    best_err = 1e18
    for name, vals in approxs.items():
        err = (vals[i] - pi) / pi * 100
        print(f" {err:19.4f}%", end='')
        if abs(err) < abs(best_err):
            best_err = err
    print(f" {best_err:9.4f}%")
