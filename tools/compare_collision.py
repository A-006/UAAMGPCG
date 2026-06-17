#!/usr/bin/env python3
"""Compare our head-on ring collision vs the AUTHOR's code on the SAME shared IC.

Ours:    output_coll_cmp/frame_*.vtk   (vorticity_magnitude, per-length ×1/dx)
Authors: /tmp/lfm_headless/coll/v{x,y,z}_*.npy (cell-centered velocity, 128x256x256)
         → curl computed here with the REAL grid spacing so |omega| is in the SAME
           per-length units as ours (apples-to-apples magnitude, not just structure).

Frames are matched by PHYSICAL TIME, not frame index (the two runs dump at different
cadences).  Produces /tmp/coll_compare.png and prints max|omega| vs time for both.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DX = 0.5 / 128  # uniform: x=0.5/128 and y=z=1/256 are both 0.00390625


def read_our_vort(path):
    L = open(path).readlines()
    dims = None
    for i, l in enumerate(L):
        t = l.split()
        if t and t[0] == "DIMENSIONS":
            dims = tuple(int(x) for x in t[1:4])
        if t and t[0] == "SCALARS" and len(t) > 1 and t[1] == "vorticity_magnitude":
            n = dims[0] * dims[1] * dims[2]
            v = np.array(L[i + 2:i + 2 + n], dtype=float).reshape((dims[2], dims[1], dims[0]))
            return np.transpose(v, (2, 1, 0))  # (x,y,z)
    raise RuntimeError("no vorticity in " + path)


def author_vort(N):
    d = "/tmp/lfm_headless/coll"
    vx = np.load(f"{d}/vx_{N:04d}.npy")  # (128,256,256) cell-centered
    vy = np.load(f"{d}/vy_{N:04d}.npy")
    vz = np.load(f"{d}/vz_{N:04d}.npy")
    wx = np.gradient(vz, DX, axis=1) - np.gradient(vy, DX, axis=2)
    wy = np.gradient(vx, DX, axis=2) - np.gradient(vz, DX, axis=0)
    wz = np.gradient(vy, DX, axis=0) - np.gradient(vx, DX, axis=1)
    return np.sqrt(wx * wx + wy * wy + wz * wz)  # (x,y,z), per-length


# time maps:  ours t = F*0.003 (51 frames over t∈[0,0.15]);  author t = N*5e-4 (saved every 25)
our_frame  = lambda t: int(round(t / 0.003))
auth_frame = lambda t: int(round(t / 5e-4 / 25.0) * 25)

TIMES = [0.05, 0.10, 0.15]

fig, ax = plt.subplots(2, 3, figsize=(13, 9), facecolor='k')
print("  t      OUR max|w|   AUTHOR max|w|   ratio (our/author)")
for c, t in enumerate(TIMES):
    ov = read_our_vort(f"output_coll_cmp/frame_{our_frame(t):05d}.vtk")
    aN = auth_frame(t)
    av = author_vort(aN)
    om = ov.max(axis=0)  # MIP down collision axis x → y-z plane (radial roll-up / filaments)
    am = av.max(axis=0)
    print(f"{t:5.2f}   {ov.max():9.1f}   {av.max():11.1f}   {ov.max()/max(av.max(),1e-9):5.2f}")
    for r, (m, nm) in enumerate([(om, "OURS"), (am, "AUTHOR")]):
        ax[r, c].imshow(m.T, origin="lower", cmap="inferno", vmax=np.percentile(m, 99.5))
        ax[r, c].set_title(f"{nm}  t={t:.2f}  max|w|={m.max():.0f}", color='w', fontsize=10)
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
plt.suptitle("Head-on ring collision: OURS vs AUTHOR (SAME IC & params) — |omega| MIP down collision axis",
             color='w')
plt.tight_layout()
plt.savefig("/tmp/coll_compare.png", dpi=80, facecolor='k')
print("\nwrote /tmp/coll_compare.png")
