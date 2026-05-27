#!/usr/bin/env python3
"""
Comprehensive Karman vortex street validation against OpenFOAM icoFoam.

Numerical:
  - Strouhal number (FFT of v-probe at x=2.0, y=0.5)
  - Wake centerline u-velocity
  - Cross-stream u-profiles at x/D stations
  - Vorticity cross-section at x=2.0

Visual:
  - Side-by-side velocity + streamlines at matched times
  - Side-by-side vorticity + contours at matched times

Usage:
  python3 validate_karman.py --our <vtk_dir> --of <of_vtk_dir> [--force-csv <csv>]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os, sys, re, glob, argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'karman'))

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════
DOMAIN_LX, DOMAIN_LY = 4.0, 1.0
CYL_CX, CYL_CY, CYL_R = 1.0, 0.5, 0.1
D, U_INF = 0.2, 1.0
VORT_CLIP = 12.0

# ═══════════════════════════════════════════════════════════════════════
# VTK readers
# ═══════════════════════════════════════════════════════════════════════

def read_vtk_ascii(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    dims = None; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('DIMENSIONS'): dims = [int(x) for x in line.split()[1:4]]
        elif line.startswith('POINT_DATA'): break
        i += 1
    nx, ny, nz = dims; data = {}; i += 1
    while i < len(lines):
        line = lines[i].strip()
        if not line: i += 1; continue
        if line.startswith('VECTORS'):
            name = line.split()[1]; vecs = np.zeros((ny, nx, 3)); i += 1
            for j in range(ny):
                for k in range(nx):
                    vals = [float(x) for x in lines[i].split()]
                    vecs[j,k] = vals[:3]; i += 1
            data[name] = vecs
        elif line.startswith('SCALARS'):
            name = line.split()[1]; scal = np.zeros((ny, nx)); i += 2
            for j in range(ny):
                for k in range(nx): scal[j,k] = float(lines[i].strip()); i += 1
            data[name] = scal
        else: i += 1
    return data, nx, ny


def load_our_frames(vtk_dir, max_frames=200):
    files = sorted(glob.glob(os.path.join(vtk_dir, 'frame_*.vtk')))
    if not files:
        raise FileNotFoundError(f"No frame_*.vtk in {vtk_dir}")
    if len(files) > max_frames:
        step = len(files) // max_frames; files = files[::step]

    dt_frame = 0.0078125 * 8
    frames = []
    for fp in files:
        data, nx, ny = read_vtk_ascii(fp)
        vel = data['velocity']; u, v = vel[:,:,0], vel[:,:,1]
        w = data.get('vorticity')
        if w is None:
            dx = 4.0/(nx-1)
            w = np.zeros_like(u)
            w[:,1:-1] += (v[:,2:]-v[:,:-2])/(2*dx)
            w[1:-1,:] -= (u[2:,:]-u[:-2,:])/(2*dx)
        fn = int(re.search(r'frame_(\d+)', os.path.basename(fp)).group(1))
        frames.append(dict(t=fn*dt_frame, u=u, v=v, w=w,
                           speed=np.sqrt(u**2+v**2), nx=nx, ny=ny))
    return frames


def load_of_frames(vtk_dir, max_frames=300):
    from of_vtk_reader import load_of_time_series
    return load_of_time_series(vtk_dir, max_frames)


def find_best_frame(frames, t_target):
    """Find frame closest to t_target."""
    idx = np.argmin([abs(f['t'] - t_target) for f in frames])
    return frames[idx], idx


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_strouhal(frames, label, probe_x=2.0, probe_y=0.5):
    """Strouhal number from FFT of v-velocity at probe point."""
    nx, ny = frames[0]['nx'], frames[0]['ny']
    dx = DOMAIN_LX / (nx - 1)
    pi = min(int(probe_x / dx), nx - 1)
    pj = min(int(probe_y / dx), ny - 1)

    times = np.array([f['t'] for f in frames])
    v_probe = np.array([f['v'][pj, pi] for f in frames])

    if len(v_probe) < 16:
        return None, None, (None, None)

    # Detrend
    v_detrend = v_probe - np.mean(v_probe)
    t_vals = np.arange(len(v_detrend))
    A = np.vstack([t_vals, np.ones_like(t_vals)]).T
    slope, intercept = np.linalg.lstsq(A, v_detrend, rcond=None)[0]
    v_detrend = v_detrend - (slope * t_vals + intercept)

    n_fft = max(len(v_detrend) * 4, 256)
    dt = np.mean(np.diff(times))
    freqs = np.fft.rfftfreq(n_fft, dt)
    spectrum = np.abs(np.fft.rfft(v_detrend, n=n_fft))

    mask = (freqs >= 0.3) & (freqs <= 3.0)
    if not np.any(mask):
        return None, None, (freqs, spectrum)

    idx_peak = np.argmax(spectrum[mask])
    f_peak = freqs[mask][idx_peak]
    St = f_peak * D / U_INF
    return f_peak, St, (freqs, spectrum)


def centerline_velocity(frame):
    nx, ny = frame['nx'], frame['ny']
    jc = ny // 2
    dx = DOMAIN_LX / (nx - 1)
    x = np.arange(nx) * dx
    return x, frame['u'][jc, :]


def cross_stream_profiles(frame, x_stations=[1.5, 2.0, 2.5, 3.0, 3.5]):
    nx, ny = frame['nx'], frame['ny']
    dx = DOMAIN_LX / (nx - 1)
    y = np.arange(ny) * dx
    profs = {}
    for xs in x_stations:
        xi = min(int(xs / dx), nx - 1)
        profs[xs] = frame['u'][:, xi]
    return y, profs


def vorticity_profile(frame, x_station=2.0):
    nx, ny = frame['nx'], frame['ny']
    dx = DOMAIN_LX / (nx - 1)
    y = np.arange(ny) * dx
    xi = min(int(x_station / dx), nx - 1)
    return y, frame['w'][:, xi]


# ═══════════════════════════════════════════════════════════════════════
# Visual comparison plots
# ═══════════════════════════════════════════════════════════════════════

def plot_side_by_side(of_frame, our_frame, t_target, output_path, dpi=150):
    """4-panel comparison: OF/Vel, Our/Vel, OF/Vort, Our/Vort."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(22, 10))
    fig.patch.set_facecolor('#0a0a0f')
    (ax_of_vel, ax_our_vel), (ax_of_vort, ax_our_vort) = axes

    for ax in axes.flat:
        ax.set_facecolor('#0a0a0f')

    # Subsampling for streamlines
    def get_stream_subsample(nx, ny):
        s = max(1, nx // 60)
        xi, yi = np.meshgrid(np.arange(0, nx, s), np.arange(0, ny, s))
        return s, xi, yi

    def get_grid(nx, ny):
        dx = DOMAIN_LX / (nx - 1)
        x = np.linspace(0, DOMAIN_LX, nx)
        y = np.linspace(0, DOMAIN_LY, ny)
        X, Y = np.meshgrid(x, y)
        return X, Y, x, y, dx

    def plot_velocity_panel(ax, frame, label, color):
        nx, ny = frame['nx'], frame['ny']
        X, Y, x1d, y1d, dx = get_grid(nx, ny)
        s, xi_s, yi_s = get_stream_subsample(nx, ny)

        ax.pcolormesh(X, Y, frame['speed'], shading='auto', cmap='bone',
                      vmin=0, vmax=1.5, rasterized=True)

        u_bulk = np.mean(frame['u'])
        u_pert = frame['u'] - u_bulk
        ax.streamplot(x1d[::s], y1d[::s],
                      u_pert[yi_s, xi_s], frame['v'][yi_s, xi_s],
                      color='lime', linewidth=0.5, density=2.5, arrowsize=0.6)

        ax.add_patch(Circle((CYL_CX, CYL_CY), CYL_R, fill=True, facecolor='#333',
                            edgecolor='white', linewidth=2.5))
        ax.axhline(y=CYL_CY, color='white', linewidth=0.5, linestyle=':', alpha=0.3)

        ax.set_title(f'{label} — Velocity    t = {frame["t"]:.2f} s',
                     fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(0, DOMAIN_LX); ax.set_ylim(0, DOMAIN_LY)
        ax.set_aspect(1.0)
        ax.set_xlabel('x', color='#aaa'); ax.set_ylabel('y', color='#aaa')
        ax.tick_params(colors='#aaa')

    def plot_vorticity_panel(ax, frame, label, color):
        nx, ny = frame['nx'], frame['ny']
        X, Y, _, _, _ = get_grid(nx, ny)

        w_clip = np.clip(frame['w'], -VORT_CLIP, VORT_CLIP)
        ax.pcolormesh(X, Y, w_clip, shading='auto', cmap='RdBu_r',
                      vmin=-VORT_CLIP, vmax=VORT_CLIP, rasterized=True)

        ax.contour(X, Y, frame['w'], levels=[2, 3, 4, 6, 9],
                   colors='#ff3333', linewidths=1.5, alpha=0.8)
        ax.contour(X, Y, frame['w'], levels=[-9, -6, -4, -3, -2],
                   colors='#3388ff', linewidths=1.5, linestyles='--', alpha=0.8)

        ax.add_patch(Circle((CYL_CX, CYL_CY), CYL_R, fill=True, facecolor='#333',
                            edgecolor='white', linewidth=2.5))

        ax.set_title(f'{label} — Vorticity $\\pm${VORT_CLIP:.0f}    t = {frame["t"]:.2f} s',
                     fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(0, DOMAIN_LX); ax.set_ylim(0, DOMAIN_LY)
        ax.set_aspect(1.0)
        ax.set_xlabel('x', color='#aaa')
        ax.tick_params(colors='#aaa')

    plot_velocity_panel(ax_of_vel, of_frame, 'OpenFOAM icoFoam', '#ff6b6b')
    plot_velocity_panel(ax_our_vel, our_frame, 'UAAMGPCG', '#4ecdc4')
    plot_vorticity_panel(ax_of_vort, of_frame, 'OpenFOAM icoFoam', '#ff6b6b')
    plot_vorticity_panel(ax_our_vort, our_frame, 'UAAMGPCG', '#4ecdc4')

    fig.suptitle(f'Karman Vortex Street — UAAMGPCG vs OpenFOAM icoFoam  |  Re=200  |  '
                 f'Matched at t≈{t_target:.1f}s',
                 fontsize=15, fontweight='bold', color='white', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor='#0a0a0f', edgecolor='none')
    plt.close(fig)
    print(f"  Saved comparison: {output_path}")


def plot_validation_metrics(results, output_path, dpi=150):
    """4-panel numerical comparison: St, wake CL, cross-stream, vort profile."""
    our = results['our']; of = results['of']

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor('#0a0a0f')
    (ax_st, ax_cl), (ax_cs, ax_vort) = axes

    for ax in axes.flat:
        ax.set_facecolor('#0a0a0f')

    # ── Panel 1: Strouhal (FFT) ──
    for key, label, color in [('of', 'OpenFOAM icoFoam', '#ff6b6b'),
                               ('our', 'UAAMGPCG', '#4ecdc4')]:
        fft_data = results[key].get('fft_data', (None, None))
        st_val = results[key].get('St')
        if fft_data[0] is not None:
            freqs, spec = fft_data
            St_axis = freqs * D / U_INF
            ax_st.semilogy(St_axis, spec / spec.max(), color=color, linewidth=2, alpha=0.9,
                          label=f'{label} (St={st_val:.4f})' if st_val else label)
    ax_st.axvline(x=0.198, color='white', linewidth=1, linestyle='--', alpha=0.4, label='St=0.198 (ref)')
    ax_st.set_xlim(0.05, 0.5)
    ax_st.set_xlabel('Strouhal Number (f D / U)', color='#aaa')
    ax_st.set_ylabel('Normalized |FFT(v)|', color='#aaa')
    ax_st.set_title('Vortex Shedding Frequency (FFT)', fontsize=13, fontweight='bold', color='white')
    ax_st.legend(fontsize=9, loc='upper right')
    ax_st.grid(True, alpha=0.15, color='white'); ax_st.tick_params(colors='#aaa')

    # ── Panel 2: Wake centerline velocity ──
    for key, label, color, ls in [('of', 'OpenFOAM', '#ff6b6b', '-'),
                                    ('our', 'UAAMGPCG', '#4ecdc4', '--')]:
        cl = results[key].get('cl_velocity')
        if cl is not None:
            x_cl, u_cl = cl
            ax_cl.plot(x_cl, u_cl, color=color, linewidth=1.8, linestyle=ls, alpha=0.9, label=label)
    ax_cl.axvline(x=CYL_CX + CYL_R, color='gray', linewidth=1, linestyle=':', alpha=0.4, label='Cylinder edge')
    ax_cl.axhline(y=U_INF, color='white', linewidth=0.5, linestyle=':', alpha=0.3)
    min_x = max(CYL_CX + CYL_R + 0.1, 1.2)
    ax_cl.set_xlim(min_x, DOMAIN_LX)
    ax_cl.set_xlabel('x', color='#aaa'); ax_cl.set_ylabel('u (centerline y=0.5)', color='#aaa')
    ax_cl.set_title('Wake Centerline Velocity', fontsize=13, fontweight='bold', color='white')
    ax_cl.legend(fontsize=9); ax_cl.grid(True, alpha=0.15, color='white')
    ax_cl.tick_params(colors='#aaa')

    # ── Panel 3: Cross-stream profiles ──
    x_stations = [1.5, 2.0, 2.5, 3.0, 3.5]
    offset_scale = 0.3
    for key, label, marker in [('of', 'OpenFOAM', 'o'), ('our', 'UAAMGPCG', 'x')]:
        cs = results[key].get('cs_profiles')
        if cs is None: continue
        y_cs, profs = cs
        for xs in x_stations:
            if xs in profs:
                offset = (xs - 1.5) * offset_scale
                color = '#ff6b6b' if key == 'of' else '#4ecdc4'
                alpha = 0.9 if key == 'of' else 0.7
                ax_cs.plot(profs[xs] + offset, y_cs, color=color, linewidth=1.5, alpha=alpha)
    for xs in x_stations:
        offset = (xs - 1.5) * offset_scale
        ax_cs.text(offset + 1.15, 0.96, f'x={xs}', color='#aaa', fontsize=7, ha='center')
    ax_cs.set_xlabel('u(y) + offset', color='#aaa'); ax_cs.set_ylabel('y', color='#aaa')
    ax_cs.set_title('Velocity Cross-Stream Profiles', fontsize=13, fontweight='bold', color='white')
    ax_cs.grid(True, alpha=0.15, color='white'); ax_cs.tick_params(colors='#aaa')

    # ── Panel 4: Vorticity profile ──
    for key, label, color, ls in [('of', 'OpenFOAM', '#ff6b6b', '-'),
                                    ('our', 'UAAMGPCG', '#4ecdc4', '--')]:
        vp = results[key].get('vort_profile')
        if vp is not None:
            y_v, w_v = vp
            ax_vort.plot(y_v, w_v, color=color, linewidth=1.8, linestyle=ls, alpha=0.9, label=label)
    ax_vort.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
    ax_vort.set_xlabel('y', color='#aaa'); ax_vort.set_ylabel('Vorticity w_z', color='#aaa')
    ax_vort.set_title('Vorticity Profile at x=2.0', fontsize=13, fontweight='bold', color='white')
    ax_vort.legend(fontsize=9); ax_vort.grid(True, alpha=0.15, color='white')
    ax_vort.tick_params(colors='#aaa')

    # ── Title with metrics ──
    our_St = results['our'].get('St'); of_St = results['of'].get('St')
    St_str = ''
    if our_St and of_St:
        St_err = abs(our_St - of_St) / of_St * 100
        St_str = f'  |  St: OF={of_St:.4f}, Ours={our_St:.4f} (Δ={St_err:.1f}%)'
    fig.suptitle(f'Karman Vortex Street Validation — UAAMGPCG vs OpenFOAM  |  Re=200{St_str}',
                 fontsize=15, fontweight='bold', color='white', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor='#0a0a0f', edgecolor='none')
    plt.close(fig)
    print(f"  Saved metrics: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Karman validation vs OpenFOAM')
    parser.add_argument('--our', default='/home/liutao/github/UAAMGPCG/output_karman_backup')
    parser.add_argument('--of', default='/home/liutao/github/UAAMGPCG/tools/karman/openfoam_case/VTK')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--t-compare', type=float, nargs='+', default=[2.0, 3.0, 3.5],
                        help='Times for side-by-side comparison')
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.our) or args.our
    os.makedirs(args.output_dir, exist_ok=True)

    print("═" * 65)
    print("  Karman Vortex Street — Comprehensive Validation")
    print("═" * 65)

    # ── Load data ──
    print(f"\nLoading OpenFOAM: {args.of}")
    try:
        of_frames = load_of_frames(args.of, args.max_frames)
    except Exception as e:
        print(f"  ERROR: {e}"); of_frames = None

    print(f"Loading UAAMGPCG: {args.our}")
    try:
        our_frames = load_our_frames(args.our, args.max_frames)
    except Exception as e:
        print(f"  ERROR: {e}"); our_frames = None

    if our_frames is None and of_frames is None:
        print("No data loaded. Abort."); sys.exit(1)

    # ── Numerical Analysis (use last common time) ──
    results = {'our': {}, 'of': {}}

    if of_frames:
        print(f"\n  OpenFOAM: {len(of_frames)} frames, t=[{of_frames[0]['t']:.2f}, {of_frames[-1]['t']:.2f}]")
    if our_frames:
        print(f"  UAAMGPCG: {len(our_frames)} frames, t=[{our_frames[0]['t']:.2f}, {our_frames[-1]['t']:.2f}]")

    print("\n─── Strouhal Number ───")
    for key, frames, label in [('of', of_frames, 'OpenFOAM'), ('our', our_frames, 'UAAMGPCG')]:
        if frames is None: continue
        f_peak, St, fft_data = compute_strouhal(frames, label)
        results[key]['St'] = St
        results[key]['f_peak'] = f_peak
        results[key]['fft_data'] = fft_data
        if St:
            print(f"  [{label}] f={f_peak:.4f} Hz, St={St:.4f} (expected 0.195-0.20)")

    # Use last frame from each for profile comparisons
    of_last = of_frames[-1] if of_frames else None
    our_last = our_frames[-1] if our_frames else None

    if of_last:
        results['of']['last_frame'] = of_last
        x_cl, u_cl = centerline_velocity(of_last)
        results['of']['cl_velocity'] = (x_cl, u_cl)
        y_cs, profs = cross_stream_profiles(of_last)
        results['of']['cs_profiles'] = (y_cs, profs)
        y_v, w_v = vorticity_profile(of_last)
        results['of']['vort_profile'] = (y_v, w_v)

    if our_last:
        results['our']['last_frame'] = our_last
        x_cl, u_cl = centerline_velocity(our_last)
        results['our']['cl_velocity'] = (x_cl, u_cl)
        y_cs, profs = cross_stream_profiles(our_last)
        results['our']['cs_profiles'] = (y_cs, profs)
        y_v, w_v = vorticity_profile(our_last)
        results['our']['vort_profile'] = (y_v, w_v)

    # ── Numerical metrics plot ──
    print("\n─── Numerical Comparison Plot ───")
    metrics_path = os.path.join(args.output_dir, 'karman_validation_metrics.png')
    plot_validation_metrics(results, metrics_path, dpi=args.dpi)

    # ── Visual comparison at matched times ──
    print("\n─── Visual Comparison at Matched Times ───")
    for t_target in args.t_compare:
        if of_frames is None or our_frames is None:
            break
        if t_target > of_frames[-1]['t'] or t_target > our_frames[-1]['t']:
            print(f"  t={t_target:.1f}: out of range, skipping")
            continue
        of_f, of_idx = find_best_frame(of_frames, t_target)
        our_f, our_idx = find_best_frame(our_frames, t_target)
        print(f"  t={t_target:.1f}: OF frame {of_idx} (t={of_f['t']:.4f}), "
              f"Our frame {our_idx} (t={our_f['t']:.4f})")
        cmp_path = os.path.join(args.output_dir, f'karman_comparison_t{t_target:.1f}.png')
        plot_side_by_side(of_f, our_f, t_target, cmp_path, dpi=args.dpi)

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  Validation Summary")
    print(f"{'='*65}")
    our_St = results['our'].get('St')
    of_St = results['of'].get('St')
    if our_St and of_St:
        St_err = abs(our_St - of_St) / of_St * 100
        ref_St = 0.198
        our_dev = abs(our_St - ref_St) / ref_St * 100
        print(f"  Strouhal:  OF = {of_St:.4f} (Δ from ref={abs(of_St-ref_St)/ref_St*100:.1f}%)")
        print(f"             Ours = {our_St:.4f} (Δ from OF={St_err:.1f}%, Δ from ref={our_dev:.1f}%)")
        passed = St_err < 5 and our_dev < 10
        print(f"  => {'PASS' if passed else 'FAIL'} (threshold: ΔOF<5%, Δref<10%)")

    ov = results['our'].get('vort_profile')
    fv = results['of'].get('vort_profile')
    if ov and fv:
        our_pp = ov[1].max() - ov[1].min()
        of_pp = fv[1].max() - fv[1].min()
        print(f"  Vorticity p-p at x=2.0: OF={of_pp:.3f}, Ours={our_pp:.3f}")

    print(f"  Output: {args.output_dir}/")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
