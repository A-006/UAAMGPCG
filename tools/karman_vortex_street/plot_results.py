"""
Generate polished comparison figures for the Karman vortex street benchmark.

Reads VTK output from multiple solvers and creates:
  1. Vorticity comparison (all solvers side-by-side)
  2. Velocity magnitude comparison
  3. Divergence convergence over time
  4. Performance summary table
"""

import os, sys, struct, json, re
import numpy as np

# Matplotlib setup — use Agg backend for headless rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

from config import *

# ── VTK ASCII reader ──

def parse_vtk_structured_points(path: str) -> dict:
    """Parse a STRUCTURED_POINTS VTK ASCII file and return field arrays."""
    with open(path) as f:
        text = f.read()

    # Split into sections
    dims_match = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)", text)
    spac_match = re.search(r"SPACING\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)", text)
    if not dims_match or not spac_match:
        raise ValueError(f"Could not parse header of {path}")

    nx, ny, _ = int(dims_match.group(1)), int(dims_match.group(2)), int(dims_match.group(3))
    dx, dy = float(spac_match.group(1)), float(spac_match.group(2))

    def read_array(section_header: str, components: int = 1):
        """Find a SCALARS or VECTORS section and parse its data."""
        pattern = re.escape(section_header) + r"\s+.*?\n"
        match = re.search(pattern, text)
        if not match:
            return None
        start = match.end()
        # Find the next section header
        rest = text[start:]
        next_header = re.search(r"^(?:SCALARS|VECTORS|POINT_DATA)", rest, re.MULTILINE)
        end = start + next_header.start() if next_header else len(text)
        data_text = text[start:end]

        # Handle LOOKUP_TABLE
        data_text = re.sub(r"LOOKUP_TABLE\s+\w+", "", data_text)

        vals = [float(v) for v in data_text.split() if v]
        npoints = nx * ny
        if components == 3:
            arr = np.array(vals).reshape(ny, nx, 3)
        else:
            arr = np.array(vals).reshape(ny, nx)
        return arr

    vel = read_array("VECTORS velocity", 3)
    vort = read_array("SCALARS vorticity", 1)
    div = read_array("SCALARS divergence", 1)
    solid = read_array("SCALARS solid", 1)

    return {
        "nx": nx, "ny": ny, "dx": dx, "dy": dy,
        "velocity": vel,
        "vorticity": vort,
        "divergence": div,
        "solid": solid,
    }


def get_final_vtk(solver_key: str, grid_label: str) -> str:
    """Return path to the last VTK file for a solver run."""
    out_dir = os.path.join(OUTPUT_DIR, solver_key, grid_label)
    if not os.path.isdir(out_dir):
        return None
    vtk_files = sorted([f for f in os.listdir(out_dir) if f.endswith(".vtk")],
                       key=lambda s: int(s.split("_")[1].split(".")[0]))
    if not vtk_files:
        return None
    return os.path.join(out_dir, vtk_files[-1])


def get_all_vtk_files(solver_key: str, grid_label: str):
    """Return sorted list of all VTK file paths."""
    out_dir = os.path.join(OUTPUT_DIR, solver_key, grid_label)
    if not os.path.isdir(out_dir):
        return []
    files = sorted([f for f in os.listdir(out_dir) if f.endswith(".vtk")],
                   key=lambda s: int(s.split("_")[1].split(".")[0]))
    return [os.path.join(out_dir, f) for f in files]


# ── Plotting ──

def plot_vorticity_comparison(grid_label: str):
    """Side-by-side vorticity comparison for all solvers."""
    fig, axes = plt.subplots(1, len(SOLVERS), figsize=(4 * len(SOLVERS), 3.5),
                              sharex=True, sharey=True)
    fig.suptitle(f"Karman Vortex Street — Vorticity Comparison ({grid_label})",
                 fontsize=13, fontweight="bold")

    vmin, vmax = 1e9, -1e9
    data_cache = {}
    for ax, solver in zip(axes, SOLVERS):
        vtk_path = get_final_vtk(solver.key, grid_label)
        if vtk_path is None:
            ax.text(0.5, 0.5, f"{solver.label}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            continue
        data = parse_vtk_structured_points(vtk_path)
        data_cache[solver.key] = data
        vort = data["vorticity"]
        vmin = min(vmin, vort.min())
        vmax = max(vmax, vort.max())

    # Use symmetric range
    vlim = max(abs(vmin), abs(vmax))
    if vlim == 0 or np.isnan(vlim):
        vlim = 1
    levels = np.linspace(-vlim, vlim, 80)

    for ax, solver in zip(axes, SOLVERS):
        data = data_cache.get(solver.key)
        if data is None:
            continue
        vort = data["vorticity"]
        x = np.linspace(0, (data["nx"] - 1) * data["dx"], data["nx"])
        y = np.linspace(0, (data["ny"] - 1) * data["dy"], data["ny"])
        X, Y = np.meshgrid(x, y)

        cf = ax.contourf(X, Y, vort, levels=levels, cmap="RdBu_r", extend="both")

        # Overlay solid
        if data["solid"] is not None:
            solid_mask = np.ma.masked_where(data["solid"] < 0.5, data["solid"])
            ax.contourf(X, Y, solid_mask, colors=["lightgray"], levels=[0, 1])

        ax.set_title(solver.label, fontsize=11, fontweight="bold", color=solver.color)
        ax.set_aspect("equal")

    axes[0].set_ylabel("y")
    for ax in axes:
        ax.set_xlabel("x")

    if data_cache:
        # Shared colorbar
        cbar = fig.colorbar(cf, ax=axes, shrink=0.92, pad=0.02, label="Vorticity ($\\omega$)")
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((-2, 2))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"vorticity_comparison_{grid_label.replace('x', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_divergence_history(grid_label: str):
    """Plot divergence over time for all solvers."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for solver in SOLVERS:
        files = get_all_vtk_files(solver.key, grid_label)
        if not files:
            continue
        times = []
        max_divs = []
        for fpath in files:
            data = parse_vtk_structured_points(fpath)
            max_divs.append(np.abs(data["divergence"]).max())
            frame = int(os.path.basename(fpath).split("_")[1].split(".")[0])
            params = KarmanParams()
            ny = max(16, int(grid_label.split("x")[0]) // 4)
            dt = 0.5 * (params.Lx / int(grid_label.split("x")[0])) / params.U_inf
            times.append(frame * params.frame_skip * dt)

        ax.semilogy(times, max_divs, "-o", color=solver.color, label=solver.label,
                    markersize=4, linewidth=1.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max |divergence|")
    ax.set_title(f"Max Divergence over Time — {grid_label}")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=5e-3, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.02, 5e-3 * 1.3, "5e-3 threshold", fontsize=8, color="gray", va="bottom")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"divergence_history_{grid_label.replace('x', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_performance_table():
    """Create a performance comparison table figure from results.json."""
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    if not os.path.exists(results_path):
        print("  No results.json found — skip performance table")
        return

    with open(results_path) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    columns = ["Solver", "Grid", "Time (s)", "ms/step", "Speedup", "max|div|"]
    cell_text = []
    baseline = None
    for grid in GRIDS:
        for solver in SOLVERS:
            matching = [r for r in results
                        if r.get("solver") == solver.key and r.get("grid") == grid.label
                        and "error" not in r]
            if not matching:
                continue
            r = matching[0]
            t = r["wall_time"]
            if solver.key == "cg" and baseline is None:
                baseline = t
            speedup = f"{baseline / t:.1f}x" if baseline and t > 0 else "—"
            cell_text.append([
                solver.label,
                grid.label,
                f"{t:.2f}",
                f"{r['ms_per_step']:.1f}",
                speedup,
                f"{r['final_div']:.2e}",
            ])

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.17, 0.12, 0.15, 0.15, 0.15, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color solver rows
    for i, row in enumerate(cell_text):
        for s in SOLVERS:
            if row[0] == s.label:
                for j in range(len(columns)):
                    table[i + 1, j].set_facecolor(f"{s.color}20")
                break

    ax.set_title("Karman Vortex Street — Solver Performance Comparison",
                 fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "performance_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_velocity_comparison(grid_label: str):
    """Side-by-side velocity magnitude + streamlines for all solvers."""
    fig, axes = plt.subplots(1, len(SOLVERS), figsize=(4 * len(SOLVERS), 3.5),
                              sharex=True, sharey=True)
    fig.suptitle(f"Karman Vortex Street — Velocity Magnitude ({grid_label})",
                 fontsize=13, fontweight="bold")

    vmax = 0
    data_cache = {}
    for ax, solver in zip(axes, SOLVERS):
        vtk_path = get_final_vtk(solver.key, grid_label)
        if vtk_path is None:
            continue
        data = parse_vtk_structured_points(vtk_path)
        data_cache[solver.key] = data
        vel = data["velocity"]
        speed = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
        vmax = max(vmax, speed.max())

    for ax, solver in zip(axes, SOLVERS):
        data = data_cache.get(solver.key)
        if data is None:
            ax.text(0.5, 0.5, f"{solver.label}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            continue
        vel = data["velocity"]
        speed = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
        x = np.linspace(0, (data["nx"] - 1) * data["dx"], data["nx"])
        y = np.linspace(0, (data["ny"] - 1) * data["dy"], data["ny"])
        X, Y = np.meshgrid(x, y)

        im = ax.imshow(speed.T, origin="lower", extent=[0, x[-1], 0, y[-1]],
                       cmap="inferno", aspect="equal", vmin=0, vmax=vmax)

        # Subsample velocity vectors
        skip = max(1, data["nx"] // 50)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  vel[::skip, ::skip, 0], vel[::skip, ::skip, 1],
                  scale=vmax * 3 + 1e-3, width=0.0015, alpha=0.5, color="white")

        # Solid
        if data["solid"] is not None:
            solid_mask = np.ma.masked_where(data["solid"] < 0.5, data["solid"])
            ax.contourf(X, Y, solid_mask, colors=["lightgray"], levels=[0, 1])

        ax.set_title(solver.label, fontsize=11, fontweight="bold", color=solver.color)
        ax.set_aspect("equal")

    axes[0].set_ylabel("y")
    for ax in axes:
        ax.set_xlabel("x")

    if data_cache:
        cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02, label="|velocity|")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"velocity_comparison_{grid_label.replace('x', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Main ──

def plot_all(grid_label: str = None):
    """Generate all figures for the given grid, or all grids if None."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    grids_to_plot = [grid_label] if grid_label else [g.label for g in GRIDS]

    for gl in grids_to_plot:
        print(f"\nPlotting for grid {gl} ...")
        plot_vorticity_comparison(gl)
        plot_velocity_comparison(gl)
        plot_divergence_history(gl)

    print(f"\nPlotting performance table ...")
    plot_performance_table()

    print(f"\nAll figures saved to {FIGURES_DIR}/")

if __name__ == "__main__":
    # Accept optional grid label from command line
    grid_label = sys.argv[1] if len(sys.argv) > 1 else None
    plot_all(grid_label)
