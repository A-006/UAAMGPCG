#!/usr/bin/env python3
"""
Karman vortex street — velocity + vortex core GIF animation (OpenFOAM style).

Shows upper vortex (top-shed, CW, -w) and lower vortex (bottom-shed, CCW, +w)
in two stacked panels: velocity magnitude + streamlines, and vorticity + contours.
Styling references OpenFOAM icoFoam visualization conventions.

Usage:
  python3 plot_karman_vortices.py [vtk_dir] [--frame N] [--gif] [--fps 8]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import os, sys, re, glob, argparse


# ═══════════════════════════════════════════════════════════════════════
# VTK reader
# ═══════════════════════════════════════════════════════════════════════

def read_vtk_ascii(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    dims = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('DIMENSIONS'):
            dims = [int(x) for x in line.split()[1:4]]
        elif line.startswith('POINT_DATA'):
            break
        i += 1

    nx, ny, nz = dims
    data = {}
    i += 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith('VECTORS'):
            name = line.split()[1]
            vecs = np.zeros((ny, nx, 3))
            i += 1
            for j in range(ny):
                for k in range(nx):
                    vals = [float(x) for x in lines[i].split()]
                    vecs[j, k] = vals[:3]
                    i += 1
            data[name] = vecs
        elif line.startswith('SCALARS'):
            name = line.split()[1]
            scal = np.zeros((ny, nx))
            i += 2
            for j in range(ny):
                for k in range(nx):
                    scal[j, k] = float(lines[i].strip())
                    i += 1
            data[name] = scal
        else:
            i += 1
    return data, nx, ny


# ═══════════════════════════════════════════════════════════════════════
# Vortex core detection
# ═══════════════════════════════════════════════════════════════════════

def find_vortex_cores(w, x_arr, y_arr, dx, min_strength=0.8, x_start=1.25):
    """Find vortex cores via centerline zero-crossings of vorticity."""
    ny, nx = w.shape
    jc = ny // 2
    x0 = int(x_start / dx)

    w_center = w[jc, x0:]
    signs = np.sign(w_center)
    crossings = []
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] and signs[i] != 0:
            crossings.append(x0 + i)

    if len(crossings) < 2:
        return []

    cores = []
    search_indices = [x0] + crossings + [nx - 1]
    for k in range(len(search_indices) - 1):
        i_start = search_indices[k]
        i_end = search_indices[k + 1]
        if i_end - i_start < 2:
            continue

        segment = w[:, i_start:i_end]
        abs_seg = np.abs(segment)
        if abs_seg.max() < min_strength:
            continue

        j_peak, i_peak = np.unravel_index(abs_seg.argmax(), segment.shape)
        i_global = i_start + i_peak

        best_j, best_i, best_aw = j_peak, i_global, abs_seg[j_peak, i_peak]
        for dj in [-3, -2, -1, 1, 2, 3]:
            jj = jc + dj
            if 0 <= jj < ny:
                lo = max(i_global - 3, 0)
                hi = min(i_global + 4, nx)
                local_abs = np.abs(w[jj, lo:hi])
                if local_abs.size > 0 and local_abs.max() > best_aw:
                    best_aw = local_abs.max()
                    li = local_abs.argmax()
                    best_j = jj
                    best_i = lo + li

        if best_aw >= min_strength:
            cores.append(dict(x=x_arr[best_i], y=y_arr[best_j],
                              w=w[best_j, best_i], aw=best_aw))

    cores.sort(key=lambda c: c['x'])
    return cores


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

VORT_CLIP = 12.0
DOMAIN_LX = 4.0
DOMAIN_LY = 1.0
CYL_CX, CYL_CY, CYL_R = 1.0, 0.5, 0.1


# ═══════════════════════════════════════════════════════════════════════
# Load all frames
# ═══════════════════════════════════════════════════════════════════════

def load_all_frames(vtk_dir, max_frames=200):
    """Load all VTK frames and precompute vortex cores, coordinates."""
    vtk_files = sorted(glob.glob(os.path.join(vtk_dir, 'frame_*.vtk')))
    if not vtk_files:
        raise FileNotFoundError(f"No frame_*.vtk files found in {vtk_dir}")

    if len(vtk_files) > max_frames:
        step = len(vtk_files) // max_frames
        vtk_files = vtk_files[::step]

    print(f"Loading {len(vtk_files)} frames...")

    all_data = []
    for fp in vtk_files:
        try:
            data, nx, ny = read_vtk_ascii(fp)
        except Exception as e:
            print(f"  Skipping {os.path.basename(fp)}: {e}")
            continue

        vel = data.get('velocity')
        w = data.get('vorticity')
        if vel is None or w is None:
            continue

        u, v = vel[:, :, 0], vel[:, :, 1]
        fn = int(re.search(r'frame_(\d+)', os.path.basename(fp)).group(1))

        all_data.append(dict(
            fn=fn, u=u, v=v, w=w, speed=np.sqrt(u**2 + v**2),
            nx=nx, ny=ny
        ))

    if not all_data:
        raise ValueError(f"No valid VTK frames found in {vtk_dir}")

    print(f"  Loaded {len(all_data)} frames, grid {nx}x{ny}")
    return all_data


# ═══════════════════════════════════════════════════════════════════════
# Animation
# ═══════════════════════════════════════════════════════════════════════

def make_animation(all_data, output_path, fps=8, dpi=120):
    """Create a 2-panel GIF: velocity (top) + vorticity (bottom)."""
    nx, ny = all_data[0]['nx'], all_data[0]['ny']
    dx = DOMAIN_LX / (nx - 1)
    x = np.linspace(0, DOMAIN_LX, nx)
    y = np.linspace(0, DOMAIN_LY, ny)
    X, Y = np.meshgrid(x, y)

    skip_s = max(1, nx // 60)
    xi_s, yi_s = np.meshgrid(np.arange(0, nx, skip_s), np.arange(0, ny, skip_s))
    x1d = np.linspace(0, DOMAIN_LX, nx)
    y1d = np.linspace(0, DOMAIN_LY, ny)

    # Precompute vortex cores & bulk flow
    print("Detecting vortex cores for all frames...")
    u_bulk = np.mean([np.mean(d['u']) for d in all_data])
    all_cores = [find_vortex_cores(d['w'], x, y, dx) for d in all_data]
    dt_est = 0.0078125 * 8  # dt * frame_skip

    # ── Setup figure ──
    plt.style.use('dark_background')
    fig, (ax_vel, ax_vort) = plt.subplots(2, 1, figsize=(20, 8))
    fig.patch.set_facecolor('#0a0a0f')

    def animate(frame_idx):
        d = all_data[frame_idx]
        cores = all_cores[frame_idx]
        t_val = d['fn'] * dt_est
        speed = d['speed']
        u, v = d['u'], d['v']
        w = d['w']

        # Circulation asymmetry
        ccw_mask = w > 0
        cw_mask = w < 0
        circ_ccw = np.sum(np.abs(w[ccw_mask]))
        circ_cw = np.sum(np.abs(w[cw_mask]))
        tot = circ_ccw + circ_cw
        asym = (circ_ccw - circ_cw) / tot * 100 if tot > 1e-10 else 0

        n_ccw = len([c for c in cores if c['w'] > 0 and c['aw'] > 1.5])
        n_cw = len([c for c in cores if c['w'] < 0 and c['aw'] > 1.5])

        # ── Panel 1: Velocity ──
        ax_vel.clear()
        ax_vel.set_facecolor('#0a0a0f')

        ax_vel.pcolormesh(X, Y, speed, shading='auto', cmap='bone',
                          vmin=0, vmax=1.5, rasterized=True)

        u_pert = u - u_bulk
        ax_vel.streamplot(x1d[::skip_s], y1d[::skip_s],
                          u_pert[yi_s, xi_s], v[yi_s, xi_s],
                          color='lime', linewidth=0.5, density=2.5, arrowsize=0.6)

        ax_vel.add_patch(Circle((CYL_CX, CYL_CY), CYL_R, fill=True, facecolor='#333',
                                edgecolor='white', linewidth=3))
        ax_vel.axhline(y=CYL_CY, color='white', linewidth=0.5, linestyle=':', alpha=0.3)

        for c in cores:
            if c['aw'] < 1.5:
                continue
            is_ccw = c['w'] > 0
            marker = 'o' if is_ccw else 's'
            color = '#ff3333' if is_ccw else '#3388ff'
            ax_vel.scatter(c['x'], c['y'], c=color, s=60 + c['aw'] * 12,
                           zorder=10, marker=marker, edgecolors='white',
                           linewidth=2, alpha=0.95)

        # Legend
        ax_vel.scatter([3.7], [0.94], c='#ff3333', s=100, marker='o',
                       edgecolors='white', linewidth=2)
        ax_vel.text(3.78, 0.94, 'Lower vortex\n(Bottom-shed, CCW, +w)',
                    color='#ff4444', fontsize=7, va='center', fontweight='bold')
        ax_vel.scatter([3.7], [0.80], c='#3388ff', s=100, marker='s',
                       edgecolors='white', linewidth=2)
        ax_vel.text(3.78, 0.80, 'Upper vortex\n(Top-shed, CW, -w)',
                    color='#4488ff', fontsize=7, va='center', fontweight='bold')

        ax_vel.set_title(f'Velocity Magnitude + Vortex Cores    t = {t_val:.2f} s',
                         fontsize=13, fontweight='bold', color='white')
        ax_vel.set_xlim(0, DOMAIN_LX)
        ax_vel.set_ylim(0, DOMAIN_LY)
        ax_vel.set_xlabel('x', color='#aaa')
        ax_vel.set_ylabel('y', color='#aaa')
        ax_vel.tick_params(colors='#aaa')
        ax_vel.set_aspect(1.0)

        # ── Panel 2: Vorticity ──
        ax_vort.clear()
        ax_vort.set_facecolor('#0a0a0f')

        w_clip = np.clip(w, -VORT_CLIP, VORT_CLIP)
        ax_vort.pcolormesh(X, Y, w_clip, shading='auto', cmap='RdBu_r',
                           vmin=-VORT_CLIP, vmax=VORT_CLIP, rasterized=True)

        ax_vort.contour(X, Y, w, levels=[2, 3, 4, 6, 9], colors='#ff3333',
                        linewidths=1.5, alpha=0.8)
        ax_vort.contour(X, Y, w, levels=[-9, -6, -4, -3, -2], colors='#3388ff',
                        linewidths=1.5, linestyles='--', alpha=0.8)

        for c in cores:
            if c['aw'] < 1.5:
                continue
            marker = 'o' if c['w'] > 0 else 's'
            ax_vort.scatter(c['x'], c['y'], c='yellow', s=40 + c['aw'] * 6,
                            zorder=10, marker=marker, edgecolors='black', linewidth=1.5)

        ax_vort.add_patch(Circle((CYL_CX, CYL_CY), CYL_R, fill=True, facecolor='#333',
                                 edgecolor='white', linewidth=3))

        ax_vort.set_title(f'Vorticity $\\pm${VORT_CLIP:.0f}  |  '
                          f'O Lower = {n_ccw}  [] Upper = {n_cw}  |  Asym = {asym:+.1f}%',
                          fontsize=13, fontweight='bold', color='white')
        ax_vort.set_xlim(0, DOMAIN_LX)
        ax_vort.set_ylim(0, DOMAIN_LY)
        ax_vort.set_aspect(1.0)
        ax_vort.set_xlabel('x', color='#aaa')
        ax_vort.tick_params(colors='#aaa')

        # ── Global title ──
        fig.suptitle(f'Karman Vortex Street — UAAMGPCG Solver  |  Re=200  |  '
                     f'O {n_ccw} Lower (CCW)  +  [] {n_cw} Upper (CW)  |  '
                     f'Frame {d["fn"]}/{all_data[-1]["fn"]}',
                     fontsize=16, fontweight='bold', color='white', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

    print(f"Rendering {len(all_data)} frames to GIF...")
    ani = FuncAnimation(fig, animate, frames=len(all_data),
                        interval=1000 // fps, blit=False)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved {output_path} ({file_size:.1f} MB)")
    plt.close(fig)


def make_single_frame(all_data, frame_idx, output_dir, dpi=150):
    """Save static PNGs for a single frame (velocity + vorticity)."""
    d = all_data[frame_idx]
    nx, ny = d['nx'], d['ny']
    dx = DOMAIN_LX / (nx - 1)
    x = np.linspace(0, DOMAIN_LX, nx)
    y = np.linspace(0, DOMAIN_LY, ny)
    X, Y = np.meshgrid(x, y)

    skip_s = max(1, nx // 60)
    xi_s, yi_s = np.meshgrid(np.arange(0, nx, skip_s), np.arange(0, ny, skip_s))
    x1d = np.linspace(0, DOMAIN_LX, nx)
    y1d = np.linspace(0, DOMAIN_LY, ny)

    u_bulk = np.mean(d['u'])
    cores = find_vortex_cores(d['w'], x, y, dx)
    dt_est = 0.0078125 * 8  # dt * frame_skip
    t_val = d['fn'] * dt_est

    ccw_mask = d['w'] > 0
    cw_mask = d['w'] < 0
    circ_ccw = np.sum(np.abs(d['w'][ccw_mask]))
    circ_cw = np.sum(np.abs(d['w'][cw_mask]))
    asym = (circ_ccw - circ_cw) / (circ_ccw + circ_cw) * 100 if (circ_ccw + circ_cw) > 1e-10 else 0
    n_ccw = len([c for c in cores if c['w'] > 0 and c['aw'] > 1.5])
    n_cw = len([c for c in cores if c['w'] < 0 and c['aw'] > 1.5])

    plt.style.use('dark_background')

    # Velocity
    fig1, ax1 = plt.subplots(figsize=(20, 6))
    fig1.patch.set_facecolor('#0a0a0f')
    ax1.set_facecolor('#0a0a0f')

    ax1.pcolormesh(X, Y, d['speed'], shading='auto', cmap='bone', vmin=0, vmax=1.5, rasterized=True)
    u_pert = d['u'] - u_bulk
    ax1.streamplot(x1d[::skip_s], y1d[::skip_s], u_pert[yi_s, xi_s], d['v'][yi_s, xi_s],
                   color='lime', linewidth=0.5, density=2.5, arrowsize=0.6)
    ax1.add_patch(Circle((CYL_CX, CYL_CY), CYL_R, fill=True, facecolor='#333',
                          edgecolor='white', linewidth=3))
    ax1.axhline(y=CYL_CY, color='white', linewidth=0.5, linestyle=':', alpha=0.3)

    for c in cores:
        if c['aw'] < 1.5: continue
        marker = 'o' if c['w'] > 0 else 's'
        color = '#ff3333' if c['w'] > 0 else '#3388ff'
        ax1.scatter(c['x'], c['y'], c=color, s=60 + c['aw'] * 12,
                    zorder=10, marker=marker, edgecolors='white', linewidth=2, alpha=0.95)

    ax1.scatter([3.7], [0.94], c='#ff3333', s=100, marker='o', edgecolors='white', linewidth=2)
    ax1.text(3.78, 0.94, 'Lower vortex\n(Bottom-shed, CCW, +w)', color='#ff4444',
             fontsize=7.5, va='center', fontweight='bold')
    ax1.scatter([3.7], [0.80], c='#3388ff', s=100, marker='s', edgecolors='white', linewidth=2)
    ax1.text(3.78, 0.80, 'Upper vortex\n(Top-shed, CW, -w)', color='#4488ff',
             fontsize=7.5, va='center', fontweight='bold')

    ax1.set_title(f'Velocity Magnitude + Vortex Cores    t = {t_val:.2f} s',
                  fontsize=14, fontweight='bold', color='white')
    ax1.set_xlim(0, DOMAIN_LX); ax1.set_ylim(0, DOMAIN_LY); ax1.set_aspect(1.0)
    ax1.set_xlabel('x', color='#aaa'); ax1.set_ylabel('y', color='#aaa')
    ax1.tick_params(colors='#aaa')

    fig1.suptitle(f'Karman Vortex Street — UAAMGPCG Solver  |  Re=200  |  '
                   f'O {n_ccw} Lower (CCW)  +  [] {n_cw} Upper (CW)  |  Asym = {asym:+.1f}%',
                   fontsize=16, fontweight='bold', color='white', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    vel_path = os.path.join(output_dir, f'karman_velocity_frame{d["fn"]:05d}.png')
    fig1.savefig(vel_path, dpi=dpi, facecolor='#0a0a0f', edgecolor='none')
    print(f"Saved {vel_path}")
    plt.close(fig1)

    # Vorticity
    fig2, ax2 = plt.subplots(figsize=(20, 6))
    fig2.patch.set_facecolor('#0a0a0f')
    ax2.set_facecolor('#0a0a0f')

    w_clip = np.clip(d['w'], -VORT_CLIP, VORT_CLIP)
    ax2.pcolormesh(X, Y, w_clip, shading='auto', cmap='RdBu_r',
                   vmin=-VORT_CLIP, vmax=VORT_CLIP, rasterized=True)
    ax2.contour(X, Y, d['w'], levels=[2, 3, 4, 6, 9], colors='#ff3333', linewidths=1.5, alpha=0.8)
    ax2.contour(X, Y, d['w'], levels=[-9, -6, -4, -3, -2], colors='#3388ff',
                linewidths=1.5, linestyles='--', alpha=0.8)

    for c in cores:
        if c['aw'] < 1.5: continue
        marker = 'o' if c['w'] > 0 else 's'
        ax2.scatter(c['x'], c['y'], c='yellow', s=40 + c['aw'] * 6,
                    zorder=10, marker=marker, edgecolors='black', linewidth=1.5)

    ax2.add_patch(Circle((CYL_CX, CYL_CY), CYL_R, fill=True, facecolor='#333',
                          edgecolor='white', linewidth=3))

    ax2.set_title(f'Vorticity $\\pm${VORT_CLIP:.0f}  |  '
                  f'O Lower = {n_ccw}  [] Upper = {n_cw}  |  Asym = {asym:+.1f}%',
                  fontsize=14, fontweight='bold', color='white')
    ax2.set_xlim(0, DOMAIN_LX); ax2.set_ylim(0, DOMAIN_LY); ax2.set_aspect(1.0)
    ax2.set_xlabel('x', color='#aaa')
    ax2.tick_params(colors='#aaa')

    fig2.suptitle(f'Karman Vortex Street — UAAMGPCG Solver  |  Re=200  |  '
                  f'Red contours = CCW (lower)  |  Blue dashed = CW (upper)',
                  fontsize=16, fontweight='bold', color='white', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    vort_path = os.path.join(output_dir, f'karman_vorticity_frame{d["fn"]:05d}.png')
    fig2.savefig(vort_path, dpi=dpi, facecolor='#0a0a0f', edgecolor='none')
    print(f"Saved {vort_path}")
    plt.close(fig2)

    print(f"\n{'='*60}")
    print(f"  UAAMGPCG Karman Vortex Street — Frame {d['fn']}")
    print(f"  Lower vortices (O, CCW, +w): {n_ccw}")
    print(f"  Upper vortices ([], CW, -w):  {n_cw}")
    print(f"  Circulation asymmetry: {asym:+.2f}%")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Karman vortex street — velocity + vortex cores (OpenFOAM style)')
    parser.add_argument('vtk_dir', nargs='?',
                        default='/home/liutao/github/UAAMGPCG/output_karman_backup',
                        help='Directory containing frame_*.vtk files')
    parser.add_argument('--frame', type=int, default=-1,
                        help='Frame number to plot (default: last frame), ignored with --gif')
    parser.add_argument('--gif', action='store_true',
                        help='Generate animated GIF instead of static PNGs')
    parser.add_argument('--output', default=None,
                        help='Output GIF path (default: <vtk_dir>/karman_animation.gif)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for PNG files (static mode)')
    parser.add_argument('--fps', type=int, default=8, help='Animation FPS')
    parser.add_argument('--dpi', type=int, default=120)
    parser.add_argument('--max-frames', type=int, default=200)
    args = parser.parse_args()

    # Load all frames
    try:
        all_data = load_all_frames(args.vtk_dir, args.max_frames)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.gif:
        output = args.output or os.path.join(args.vtk_dir, 'karman_animation.gif')
        make_animation(all_data, output, fps=args.fps, dpi=args.dpi)
    else:
        if args.output_dir is None:
            args.output_dir = args.vtk_dir
        os.makedirs(args.output_dir, exist_ok=True)

        if args.frame >= 0:
            idx = next((i for i, d in enumerate(all_data) if d['fn'] == args.frame), None)
            if idx is None:
                print(f"ERROR: Frame {args.frame} not found. Available: {[d['fn'] for d in all_data]}")
                sys.exit(1)
        else:
            idx = -1

        make_single_frame(all_data, idx, args.output_dir, dpi=args.dpi)


if __name__ == '__main__':
    main()
