#!/usr/bin/env python3
"""Render a 3D LFM vortex-ring run (structured-points VTK frames) to an animated GIF.

Reads the `vorticity_magnitude` field written by VtkWriter3D. Two view modes:

  side_z (default) — a single ring propagating along +z:
    left  : maximum-intensity projection of |ω| over the y axis (side view)
    right : the x-z mid-plane slice of |ω|

  collision_x — a head-on collision of two rings coaxial with the x-axis:
    left  : x-z mid-plane slice (rings approach along x, expand along z)
    right : y-z slice at the collision plane x=Lx/2 (ring seen face-on, expanding)

Usage: render_vortex_ring.py [frames_dir] [out.gif] [fps] [mode]
  defaults: output_vortex_ring_lfm  <dir>/vortex_ring_lfm.gif  12  side_z
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
    """Parse the vorticity_magnitude scalar block from a structured-points VTK."""
    with open(path) as f:
        lines = f.readlines()
    dims = None
    spacing = None
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        if tok and tok[0] == "DIMENSIONS":
            dims = tuple(int(x) for x in tok[1:4])
        elif tok and tok[0] == "SPACING":
            spacing = tuple(float(x) for x in tok[1:4])
        elif tok and tok[0] == "SCALARS" and tok[1] == "vorticity_magnitude":
            n = dims[0] * dims[1] * dims[2]
            start = i + 2  # skip SCALARS + LOOKUP_TABLE lines
            vals = np.array(lines[start:start + n], dtype=float)
            # VTK structured points: x fastest, then y, then z → array[z, y, x]
            return vals.reshape((dims[2], dims[1], dims[0])), spacing
        i += 1
    raise RuntimeError(f"vorticity_magnitude not found in {path}")


def build_panels(files, mode):
    """Return (panelL, panelR, titles, extentL, extentR, axis_labels) for a mode."""
    A, B = [], []
    spacing = None
    for f in files:
        v, spacing = read_vorticity(f)  # v[z, y, x]
        nz, ny, nx = v.shape
        if mode == "collision_x":
            A.append(v[:, ny // 2, :])   # x-z slice (y=mid) → (z, x)
            B.append(v[:, :, nx // 2])   # y-z slice at collision plane x=mid → (z, y)
        else:  # side_z
            A.append(v.max(axis=1))      # max over y → (z, x): side view
            B.append(v[:, ny // 2, :])   # x-z mid-plane slice → (z, x)
    A, B = np.array(A), np.array(B)
    sx, sy, sz = spacing
    if mode == "collision_x":
        titles = ["|ω|  x-z mid-plane (rings collide along x)",
                  "|ω|  y-z plane at x=Lx/2 (ring expands face-on)"]
        extents = ([0, A.shape[2] * sx, 0, A.shape[1] * sz],
                   [0, B.shape[2] * sy, 0, B.shape[1] * sz])
        labels = [("x", "z"), ("y", "z")]
    else:
        titles = ["|ω|  max-projection over y (side view)", "|ω|  x-z mid-plane slice"]
        extents = ([0, A.shape[2] * sx, 0, A.shape[1] * sz],
                   [0, B.shape[2] * sx, 0, B.shape[1] * sz])
        labels = [("x", "z"), ("x", "z")]
    return A, B, titles, extents, labels


def main():
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else "output_vortex_ring_lfm"
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(frames_dir, "vortex_ring_lfm.gif")
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    mode = sys.argv[4] if len(sys.argv) > 4 else "side_z"

    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.vtk")))
    if not files:
        print(f"No frames in {frames_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Reading {len(files)} frames from {frames_dir} (mode={mode}) ...")

    A, B, titles, extents, labels = build_panels(files, mode)

    # Fixed robust color scale so the structure stays visible across the run.
    vmax = float(np.percentile(np.concatenate([A.ravel(), B.ravel()]), 99.0))

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 4.4))
    ims = []
    for a, data, title, ext, (lx, lz) in zip(ax, (A, B), titles, extents, labels):
        im = a.imshow(data[0], origin="lower", extent=ext, cmap="inferno",
                      vmin=0, vmax=vmax, aspect="equal", interpolation="bilinear")
        a.set_title(title, fontsize=9)
        a.set_xlabel(lx)
        a.set_ylabel(lz)
        ims.append(im)
    sup = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    label = "vortex ring" if mode == "side_z" else "vortex-ring collision"

    def update(k):
        ims[0].set_data(A[k])
        ims[1].set_data(B[k])
        sup.set_text(f"3D LFM {label} (GPU UAAMG Poisson) — frame {k}/{len(files) - 1}")
        return ims + [sup]

    anim = FuncAnimation(fig, update, frames=len(files), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    print(f"Wrote {out_path}  ({len(files)} frames, {fps:.0f} fps)")


if __name__ == "__main__":
    main()
