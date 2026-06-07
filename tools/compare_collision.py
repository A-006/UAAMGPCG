#!/usr/bin/env python3
"""Compare our collision vorticity vs the authors' (same shared IC).

Ours:    output_coll_ours/frame_*.vtk   (vorticity_magnitude on a 129x257x257 node grid)
Authors: /tmp/lfm_headless/coll/v{x,y,z}_*.npy (cell-centered velocity, 128x256x256)
         → vorticity computed here in numpy (curl), avoiding the tiled GetVorNorm artifact.

Produces /tmp/coll_compare.png (side-by-side volume render of the last frame) and prints
max|omega| vs time for both, plus the IC round-trip check if available.
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ea(vol, axis, vmax, lo=0.30, g=1.0, k=6):
    d = np.clip((vol / vmax - lo) / (1 - lo), 0, 1) ** g
    d = np.moveaxis(d, axis, 0)
    al = 1 - np.exp(-k * d)
    T = np.cumprod(1 - al + 1e-6, axis=0)
    T = np.concatenate([np.ones_like(T[:1]), T[:-1]], axis=0)
    img = np.sum(al * T, axis=0)
    return img / (img.max() + 1e-9)


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


def author_vort(frame):
    d = "/tmp/lfm_headless/coll"
    vx = np.load(f"{d}/vx_{frame:04d}.npy")  # (128,256,256) cell-centered
    vy = np.load(f"{d}/vy_{frame:04d}.npy")
    vz = np.load(f"{d}/vz_{frame:04d}.npy")
    # curl on cell centers (unit spacing → magnitude proportional, fine for comparison)
    wx = np.gradient(vz, axis=1) - np.gradient(vy, axis=2)
    wy = np.gradient(vx, axis=2) - np.gradient(vz, axis=0)
    wz = np.gradient(vy, axis=0) - np.gradient(vx, axis=1)
    return np.sqrt(wx * wx + wy * wy + wz * wz)


def main():
    our = sorted(glob.glob("output_coll_ours/frame_*.vtk"))
    auth = sorted(glob.glob("/tmp/lfm_headless/coll/vx_*.npy"))
    print(f"our frames: {len(our)}   author frames: {len(auth)}")
    if not our or not auth:
        print("missing frames; abort"); return

    # |omega|max vs frame
    print("\nframe   our max|w|   author max|w|(curl, ×1/dx units differ)")
    af = [int(os.path.basename(p).split('_')[1].split('.')[0]) for p in auth]
    for n in range(min(len(our), len(af))):
        ow = read_our_vort(our[n]).max()
        aw = author_vort(af[n]).max()
        print(f"{n:3d}     {ow:8.1f}     {aw:8.4f}")

    # side-by-side last frame
    ov = read_our_vort(our[-1]); av = author_vort(af[-1])
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), facecolor='k')
    for col, (vol, name) in enumerate([(ov, "OURS"), (av, "AUTHOR")]):
        vmax = float(np.percentile(vol, 99.5))
        ax[0, col].imshow(ea(vol, 0, vmax).T, origin="lower", cmap="inferno")
        ax[0, col].set_title(f"{name} face-on (down collision x)", color='w')
        ax[1, col].imshow(ea(vol, 2, vmax).T, origin="lower", cmap="inferno", aspect="auto")
        ax[1, col].set_title(f"{name} side", color='w')
        for r in (0, 1):
            ax[r, col].set_xticks([]); ax[r, col].set_yticks([])
    plt.suptitle("Collision: OURS vs AUTHOR (same IC) — last frame |omega|", color='w')
    plt.tight_layout(); plt.savefig("/tmp/coll_compare.png", dpi=72, facecolor='k')
    print("\nwrote /tmp/coll_compare.png")


if __name__ == "__main__":
    main()
