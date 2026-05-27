#!/usr/bin/env python3
"""
Wake-focused Karman animation: clip vorticity tight to highlight downstream shedding,
mask the boundary-layer spike near the cylinder.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(__file__))
from make_karman_animation import read_vtk


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'output_karman_gpu'
    files = sorted(glob.glob(os.path.join(out_dir, 'frame_*.vtk')))
    if not files:
        print(f"No frames in {out_dir}")
        return 1
    print(f"{len(files)} frames")

    frames = []
    for fp in files:
        d, dx, dy, dims = read_vtk(fp)
        m = re.search(r'frame_(\d+)', fp)
        frames.append({'data': d, 'dx': dx, 'dy': dy, 'dims': dims,
                       'frame': int(m.group(1)) if m else 0})

    nx_pts, ny_pts = frames[0]['dims'][0], frames[0]['dims'][1]
    dx, dy = frames[0]['dx'], frames[0]['dy']
    x = np.linspace(0, (nx_pts - 1) * dx, nx_pts)
    y = np.linspace(0, (ny_pts - 1) * dy, ny_pts)
    X, Y = np.meshgrid(x, y)
    Lx = (nx_pts - 1) * dx
    Ly = (ny_pts - 1) * dy

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(13, 4), facecolor='#0a0a0f')
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.10)

    cyl_cx, cyl_cy, cyl_R = 1.0, 0.5, 0.1
    VORT_CLIP = 8.0   # tight clip — highlight wake, suppress BL spike

    def render(idx):
        f = frames[idx]
        v = f['data']['vorticity']
        ax.clear()
        ax.set_facecolor('#0a0a0f')
        ax.pcolormesh(X, Y, np.clip(v, -VORT_CLIP, VORT_CLIP),
                      cmap='RdBu_r', vmin=-VORT_CLIP, vmax=VORT_CLIP,
                      shading='auto', rasterized=True)
        ax.add_patch(Circle((cyl_cx, cyl_cy), cyl_R, fill=True,
                            facecolor='#222', edgecolor='white', linewidth=2))
        # Contours of the actual (not clipped) value
        ax.contour(X, Y, v, levels=[3, 6], colors='#ffb86b', linewidths=0.6, alpha=0.7)
        ax.contour(X, Y, v, levels=[-6, -3], colors='#6bdcff',
                   linewidths=0.6, linestyles='--', alpha=0.7)
        t_sim = f['frame'] * 20 * 0.0039  # frame_skip * cycle_dt
        ax.set_title(f"LFM Karman vortex street   Re=200   256×64   "
                     f"frame {idx+1}/{len(frames)}   t≈{t_sim:.1f}s",
                     color='white', fontsize=12)
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_aspect('equal')
        ax.tick_params(colors='#aaa', labelsize=8)

    ani = FuncAnimation(fig, render, frames=len(frames), interval=120, blit=False)
    gif_path = os.path.join(out_dir, 'karman_wake.gif')
    ani.save(gif_path, writer=PillowWriter(fps=8), dpi=90)
    print(f"Wrote {gif_path} ({os.path.getsize(gif_path)/1024/1024:.2f} MB)")


if __name__ == '__main__':
    main()
