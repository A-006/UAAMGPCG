#!/usr/bin/env python3
"""Apples-to-apples collision trajectory: OURS vs AUTHOR on the SAME IC (/tmp/coll_ic),
same vorticity operator (np.gradient on cell-centered velocity).

OURS:    output_coll_match/v{x,y,z}_NNNNN.raw  (float32, 128x256x256, C-order i-slowest)
         frame f dumped at cycle 25f → t = f*0.0125
AUTHOR:  /tmp/lfm_headless/coll/v{x,y,z}_NNNN.npy  (float32, 128x256x256)
         file index N at t = N*5e-4   →  for t=f*0.0125, N = f*25
"""
import numpy as np, os, sys

DX = 0.5 / 128
NX, NY, NZ = 128, 256, 256


def curlmax(vx, vy, vz):
    wx = np.gradient(vz, DX, axis=1) - np.gradient(vy, DX, axis=2)
    wy = np.gradient(vx, DX, axis=2) - np.gradient(vz, DX, axis=0)
    wz = np.gradient(vy, DX, axis=0) - np.gradient(vx, DX, axis=1)
    return np.sqrt(wx * wx + wy * wy + wz * wz)


def ours(f):
    d = "output_coll_match"
    try:
        vx = np.fromfile(f"{d}/vx_{f:05d}.raw", dtype=np.float32).reshape(NX, NY, NZ)
        vy = np.fromfile(f"{d}/vy_{f:05d}.raw", dtype=np.float32).reshape(NX, NY, NZ)
        vz = np.fromfile(f"{d}/vz_{f:05d}.raw", dtype=np.float32).reshape(NX, NY, NZ)
    except FileNotFoundError:
        return None
    if vx.size != NX * NY * NZ:
        return None
    return curlmax(vx, vy, vz)


def author(N):
    d = "/tmp/lfm_headless/coll"
    try:
        vx = np.load(f"{d}/vx_{N:04d}.npy"); vy = np.load(f"{d}/vy_{N:04d}.npy"); vz = np.load(f"{d}/vz_{N:04d}.npy")
    except FileNotFoundError:
        return None
    return curlmax(vx, vy, vz)


print("  t        OURS max|w|   AUTHOR max|w|   ratio")
for f in range(0, 21):
    t = f * 0.0125
    ov = ours(f)
    av = author(f * 25)
    if ov is None and av is None:
        continue
    om = ov.max() if ov is not None else float('nan')
    am = av.max() if av is not None else float('nan')
    r = om / am if (av is not None and am > 0 and ov is not None) else float('nan')
    print(f"{t:6.4f}   {om:10.1f}   {am:11.1f}   {r:5.2f}")
