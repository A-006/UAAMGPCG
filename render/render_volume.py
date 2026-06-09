#!/usr/bin/env python3
"""Fast volumetric render of a |ω| field from structured-points VTK frames.

matplotlib's 3D Poly3DCollection chokes on the ~1e6 triangles a low iso-level
produces at 256^3, and looks crude. This instead does a pure-numpy
emission-absorption volume integration (vectorized, ~0.1 s/frame) which both runs
fast at high resolution and gives the Houdini-like volumetric look of the paper:
the head-on collision's radial secondary-vortex filaments read clearly.

Two panels per frame:
  - left : view DOWN the collision axis x  -> the ring face-on (radial "tiara")
  - right: view along y                    -> the side profile (expansion)

Usage: render_volume.py <frames_dir> <out.gif> [gamma] [fps] [cmap]
  defaults: gamma=0.7  fps=12  cmap=inferno
"""
import sys
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def read_vorticity(path):
    dims = None
    with open(path) as f:
        while True:
            ln = f.readline()
            if not ln:
                break
            t = ln.split()
            if not t:
                continue
            if t[0] == "DIMENSIONS":
                dims = tuple(int(x) for x in t[1:4])
            elif t[0] == "LOOKUP_TABLE":
                break
        if dims is None:
            raise RuntimeError(f"no DIMENSIONS in {path}")
        vals = np.fromstring(f.read(), sep="\n")
    n = dims[0] * dims[1] * dims[2]
    return vals[:n].reshape((dims[2], dims[1], dims[0]))  # [z,y,x]


def emission_absorption(vol, axis, vmax, gamma, lo=0.45, kappa=8.0):
    """Front-to-back emission-absorption integration along `axis`.
    A transfer function maps |ω| through a low floor `lo` (fraction of vmax) so
    the diffuse background goes transparent and only the sharp high-vorticity
    filaments emit; `kappa` is the per-slab opacity. Returns a 2D image in [0,1]."""
    d = np.clip((vol / vmax - lo) / (1 - lo), 0, 1) ** gamma
    d = np.moveaxis(d, axis, 0)             # integrate over leading axis
    alpha = 1.0 - np.exp(-kappa * d)        # opacity per slab
    T = np.cumprod(1.0 - alpha + 1e-6, axis=0)
    T = np.concatenate([np.ones_like(T[:1]), T[:-1]], axis=0)  # transmittance in front
    img = np.sum(alpha * T, axis=0)         # accumulated emission
    return img / (img.max() + 1e-9)


def main():
    frames_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(frames_dir, "volume.gif")
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
    cmap = sys.argv[5] if len(sys.argv) > 5 else "inferno"

    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.vtk")))
    if not files:
        print(f"No frames in {frames_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Reading {len(files)} frames from {frames_dir} ...")
    vols = [read_vorticity(f) for f in files]
    vmax = float(np.percentile(np.concatenate([v.ravel() for v in vols[::4]]), 99.5))
    print(f"render grid {vols[0].shape[::-1]}  vmax(p99.5)={vmax:.1f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.4))
    for a in (axL, axR):
        a.set_xticks([]); a.set_yticks([])
    fig.patch.set_facecolor("black")

    def draw(k):
        axL.clear(); axR.clear()
        for a in (axL, axR):
            a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
        face = emission_absorption(vols[k], axis=2, vmax=vmax, gamma=gamma)  # along x
        side = emission_absorption(vols[k], axis=1, vmax=vmax, gamma=gamma)  # along y
        axL.imshow(face, origin="lower", cmap=cmap, vmin=0, vmax=1)
        axR.imshow(side, origin="lower", cmap=cmap, vmin=0, vmax=1)
        axL.set_title("face-on  (down collision axis x)", color="w", fontsize=10)
        axR.set_title("side  (along y)", color="w", fontsize=10)
        fig.suptitle(f"|ω| volume render — frame {k}/{len(files)-1}", color="w", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=len(files), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": "black"})
    print(f"Wrote {out_path}  ({len(files)} frames, {fps:.0f} fps)")


if __name__ == "__main__":
    main()
