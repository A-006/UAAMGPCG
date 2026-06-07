#!/usr/bin/env python3
"""Render a 3D vorticity iso-surface animation from structured-points VTK frames.

Marching-cubes iso-surface of |ω| rendered with matplotlib 3D — closer to the
paper's Houdini-style 3D look than slices/projections, so the radial secondary
vortex filaments of a head-on ring collision are visible.

Usage: render_isosurface.py [frames_dir] [out.gif] [level_frac] [elev] [azim] [fps]
  defaults: <dir>  <dir>/isosurface.gif  0.18  18  -72  12
  level_frac: iso-level as a fraction of the run's 99.5th-percentile |ω|.
"""
import sys
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def downsample(vol, target_max=140):
    """Block-mean pool so the largest axis is <= target_max. Rendering only —
    keeps marching-cubes triangle counts (and matplotlib) tractable at high res,
    and smooths grid-scale noise. Physics data on disk is untouched."""
    f = max(1, int(np.ceil(max(vol.shape) / target_max)))
    if f == 1:
        return vol
    nz, ny, nx = vol.shape
    cz, cy, cx = (nz // f) * f, (ny // f) * f, (nx // f) * f
    v = vol[:cz, :cy, :cx].reshape(cz // f, f, cy // f, f, cx // f, f)
    return v.mean(axis=(1, 3, 5))


def read_vorticity(path):
    # Slurp header lines, then bulk-parse the numeric block with np.fromstring
    # (C-level, ~50x faster than per-line float() for 256^3 ASCII frames).
    dims = spacing = None
    with open(path) as f:
        data_off = None
        while True:
            ln = f.readline()
            if not ln:
                break
            t = ln.split()
            if not t:
                continue
            if t[0] == "DIMENSIONS":
                dims = tuple(int(x) for x in t[1:4])
            elif t[0] == "SPACING":
                spacing = tuple(float(x) for x in t[1:4])
            elif t[0] == "LOOKUP_TABLE":
                data_off = f.tell()  # numbers start right after this line
                break
        if data_off is None:
            raise RuntimeError(f"vorticity_magnitude not found in {path}")
        vals = np.fromstring(f.read(), sep="\n")
    n = dims[0] * dims[1] * dims[2]
    return vals[:n].reshape((dims[2], dims[1], dims[0])), spacing  # [z,y,x]


def main():
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(frames_dir, "isosurface.gif")
    level_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.18
    elev = float(sys.argv[4]) if len(sys.argv) > 4 else 18.0
    azim = float(sys.argv[5]) if len(sys.argv) > 5 else -72.0
    fps = float(sys.argv[6]) if len(sys.argv) > 6 else 12.0

    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.vtk")))
    if not files:
        print(f"No frames in {frames_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Reading {len(files)} frames from {frames_dir} ...")
    vols, sp = [], None
    for f in files:
        v, sp = read_vorticity(f)
        vols.append(downsample(v))
    print(f"render grid (after downsample): {vols[0].shape[::-1]}")
    gmax = float(np.percentile(np.concatenate([v.ravel() for v in vols[::4]]), 99.5))
    level = level_frac * gmax
    print(f"iso-level = {level:.2f}  (= {level_frac} × p99.5={gmax:.1f})")

    nz, ny, nx = vols[0].shape
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(k):
        ax.clear()
        vol = vols[k]
        try:
            verts, faces, _, _ = measure.marching_cubes(vol, level=level)
        except (ValueError, RuntimeError):
            verts = None
        if verts is not None and len(faces):
            # verts are in (z,y,x) index space → map to (x,y,z) physical
            z, y, x = verts[:, 0], verts[:, 1], verts[:, 2]
            tris = np.stack([x[faces], y[faces], z[faces]], axis=-1)
            mesh = Poly3DCollection(tris, alpha=0.9, linewidths=0)
            # shade by depth along the collision axis for a bit of form
            mesh.set_facecolor(plt.cm.plasma(0.55))
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)
        ax.set_box_aspect((nx, ny, nz))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"|ω| iso-surface — frame {k}/{len(files)-1}", fontsize=10)

    anim = FuncAnimation(fig, draw, frames=len(files), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    print(f"Wrote {out_path}  ({len(files)} frames, {fps:.0f} fps)")


if __name__ == "__main__":
    main()
