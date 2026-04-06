"""
Plot absolute difference |approx(n) - pi(n)| for various pi(n) approximations.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi

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

from scipy.special import zeta as scipy_zeta
import math

def pi_li(x):
    return np.array([expi(np.log(xi)) - expi(np.log(2)) for xi in x])

def pi_li_corrected(x):
    li_x = np.array([expi(np.log(xi)) - expi(np.log(2)) for xi in x])
    li_sqrt = np.array([expi(np.log(xi**0.5)) - expi(np.log(2)) for xi in x])
    return li_x - li_sqrt / 2

def R_gram_single(x):
    lnx = math.log(x)
    result = 1.0
    term = 1.0
    for k in range(1, 200):
        term *= lnx / k
        z = float(scipy_zeta(k + 1))
        c = term / (k * z)
        result += c
        if abs(c) < 1e-15 * abs(result):
            break
    return result

def pi_R_gram(x):
    return np.array([R_gram_single(xi) for xi in x])

approxs = {
    'n/ln(n)':             ns / np.log(ns),
    'n/(ln(n)-1)':         ns / (np.log(ns) - 1),
    'Li(x)':               pi_li(ns),
    'Li(x)-Li(√x)/2':    pi_li_corrected(ns),
    'R(x) Gram':           pi_R_gram(ns),
}

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))

# --- Plot 1: Absolute difference (log-log) ---
for (name, vals), color in zip(approxs.items(), colors):
    diff = np.abs(vals - pis)
    # avoid log(0)
    diff = np.maximum(diff, 0.5)
    ax1.loglog(ns, diff, 'o-', color=color, markersize=3, linewidth=1.5, label=name)

# also plot pi(n) itself for scale reference
ax1.loglog(ns, pis, 'k--', linewidth=0.8, alpha=0.4, label='π(n) (reference scale)')

ax1.set_xlabel('n', fontsize=12)
ax1.set_ylabel('|approx(n) − π(n)|', fontsize=12)
ax1.set_title('Absolute error of π(n) approximations', fontsize=14)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)

# --- Plot 2: Signed relative difference (semilog-x) ---
for (name, vals), color in zip(approxs.items(), colors):
    rel_diff = (vals - pis) / pis
    ax2.semilogx(ns, rel_diff, 'o-', color=color, markersize=3, linewidth=1.5, label=name)

ax2.axhline(y=0, color='black', linewidth=0.5)
# add +/- 1% lines
ax2.axhline(y=0.01, color='gray', linewidth=0.5, linestyle=':')
ax2.axhline(y=-0.01, color='gray', linewidth=0.5, linestyle=':')

ax2.set_xlabel('n', fontsize=12)
ax2.set_ylabel('(approx − π(n)) / π(n)', fontsize=12)
ax2.set_title('Signed relative error', fontsize=14)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.20, 0.20)

plt.tight_layout()
plt.savefig('c:/Users/zball/Documents/bigint/libfact/bench/pi_diff.png', dpi=150)
print("Saved to bench/pi_diff.png")
