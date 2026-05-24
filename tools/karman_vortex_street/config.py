"""
Karman Vortex Street benchmark configuration.

Defines grid resolutions, solvers, and parameters for automated comparison.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import os

@dataclass
class SolverConfig:
    """Configuration for a single solver run."""
    key: str            # Factory key (cg, pcg_gmg, pcg_amg, pcg_uaamg)
    label: str          # Display name
    iters: int          # Iterations per pressure solve
    color: str          # Plot color

# ── Solver suite ──
SOLVERS: List[SolverConfig] = [
    SolverConfig("cg",        "CG",              200, "#3498db"),  # blue
    SolverConfig("pcg_gmg",   "PCG / GMG",        50, "#2ecc71"),  # green
    SolverConfig("pcg_amg",   "PCG / AMG",        50, "#e67e22"),  # orange
    SolverConfig("pcg_uaamg", "PCG / UAAMG",      50, "#e74c3c"),  # red
]

@dataclass
class GridConfig:
    """Grid resolution configuration."""
    nx: int
    label: str

GRIDS: List[GridConfig] = [
    GridConfig(128, "128x32"),
    GridConfig(256, "256x64"),
]

# ── Physical parameters ──
@dataclass
class KarmanParams:
    """Physical parameters for the Karman vortex street."""
    Lx: float       = 4.0
    Ly: float       = 1.0
    U_inf: float    = 1.0
    Re: float       = 200.0
    cyl_cx: float   = 1.0
    cyl_cy: float   = 0.5
    cyl_R: float    = 0.1
    t_end: float    = 2.0
    solve_tol: float = 1e-6
    frame_skip: int = 8

# ── Paths (relative to this directory) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
LFM_EXE = os.path.join(PROJECT_ROOT, "build/src/2d/lfm_2d")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
