import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_rows(text: str):
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            na = int(parts[0])
            nb = int(parts[1])
            engine = parts[2]
            ntt_ns = float(parts[3])
            gmp_ns = float(parts[4])
        except ValueError:
            continue
        ratio = gmp_ns / ntt_ns if ntt_ns > 0.0 else math.nan
        rows.append(
            {
                "na": na,
                "nb": nb,
                "engine": engine,
                "ntt_ns": ntt_ns,
                "gmp_ns": gmp_ns,
                "delta_ns": gmp_ns - ntt_ns,
                "ratio": ratio,
                "balanced": na == nb,
                "size": max(na, nb),
            }
        )
    return rows


def select(rows, balanced: bool):
    return [row for row in rows if row["balanced"] == balanced]


def plot_times(ax, rows, title: str):
    x = [row["size"] for row in rows]
    ntt = [row["ntt_ns"] for row in rows]
    gmp = [row["gmp_ns"] for row in rows]

    ax.loglog(x, ntt, "o-", color="#0b6e4f", linewidth=1.8, markersize=4, label="NTT")
    ax.loglog(x, gmp, "o-", color="#c84c09", linewidth=1.8, markersize=4, label="GMP")
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Time (ns)", fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=10)


def plot_difference(ax, rows, title: str):
    x = [row["size"] for row in rows]
    delta_us = [(row["delta_ns"] / 1e3) for row in rows]
    ratio = [row["ratio"] if row["gmp_ns"] > 0.0 else math.nan for row in rows]

    ax.semilogx(x, delta_us, "o-", color="#1f4b99", linewidth=1.8, markersize=4)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Operand size (max limbs)", fontsize=11)
    ax.set_ylabel("GMP - NTT (us)", fontsize=11)
    ax.grid(True, which="both", alpha=0.25)

    twin = ax.twinx()
    twin.semilogx(x, ratio, "s--", color="#b02e0c", linewidth=1.2, markersize=3.5, label="GMP/NTT")
    twin.set_ylabel("GMP / NTT", fontsize=11, color="#b02e0c")
    twin.tick_params(axis="y", colors="#b02e0c")
    twin.axhline(1.0, color="#b02e0c", linewidth=0.8, linestyle=":", alpha=0.7)


def main():
    parser = argparse.ArgumentParser(description="Plot bench_ntt_vs_gmp output.")
    parser.add_argument("input", nargs="?", help="Input file. Reads stdin if omitted.")
    parser.add_argument(
        "--output",
        default="bench/fixint/bench_ntt_vs_gmp_diff.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    rows = parse_rows(text)
    if not rows:
        raise SystemExit("no benchmark rows found")

    balanced = select(rows, True)
    unbalanced = select(rows, False)

    plt.style.use("tableau-colorblind10")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    plot_times(axes[0][0], balanced, "Balanced Multiply Times")
    plot_times(axes[0][1], unbalanced, "Unbalanced Multiply Times")
    plot_difference(axes[1][0], balanced, "Balanced Difference")
    plot_difference(axes[1][1], unbalanced, "Unbalanced Difference")

    fig.suptitle("bench_ntt_vs_gmp: NTT vs GMP", fontsize=15)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(output_path)


if __name__ == "__main__":
    main()
