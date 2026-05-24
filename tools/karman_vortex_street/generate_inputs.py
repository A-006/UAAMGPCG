"""
Generate INPUT files for each solver and grid resolution.
"""

import os
from config import *

def make_input(grid: GridConfig, solver: SolverConfig, params: KarmanParams, out_dir: str) -> str:
    """Write an INPUT file and return its path."""
    ny = max(16, grid.nx // 4)
    dt = 0.5 * (params.Lx / grid.nx) / params.U_inf
    content = f"""scenario = karman
NX       = {grid.nx}
NY       = {ny}
Lx       = {params.Lx}
Ly       = {params.Ly}
U_inf    = {params.U_inf}
Re       = {params.Re}
cyl_cx   = {params.cyl_cx}
cyl_cy   = {params.cyl_cy}
cyl_R    = {params.cyl_R}
t_end    = {params.t_end}
dt       = {dt}
solver   = {solver.key}
solve_iters = {solver.iters}
solve_tol   = {params.solve_tol}
frame_skip  = {params.frame_skip}
out_dir  = {out_dir}
"""
    os.makedirs(INPUTS_DIR, exist_ok=True)
    fname = f"karman_{solver.key}_{grid.label.replace('x', '_')}.txt"
    path = os.path.join(INPUTS_DIR, fname)
    with open(path, "w") as f:
        f.write(content)
    return path

def generate_all(params: KarmanParams = KarmanParams()):
    """Generate INPUT files for all solver/grid combinations."""
    files = {}
    for solver in SOLVERS:
        for grid in GRIDS:
            solver_dir = os.path.join(OUTPUT_DIR, solver.key, grid.label)
            path = make_input(grid, solver, params, solver_dir)
            files[(solver.key, grid.label)] = path
            print(f"  Generated {path}")
    return files

if __name__ == "__main__":
    generate_all()
