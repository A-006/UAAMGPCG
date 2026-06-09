#!/usr/bin/env python3
"""
Karman vortex street — clearly visible vortex identification.

Vortex labeling by SHEDDING ORIGIN (not position):
  - Bottom-shed vortex (CCW, +w) -> RED circles O
  - Top-shed vortex    (CW,  -w) -> BLUE squares []

After shedding, vortices drift across the centerline.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import glob, re, os

# ══ VTK reader ══
def read_vtk(filepath):
    with open(filepath) as f: lines = f.readlines()
    dims, spacing, i = None, [1,1,1], 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('DIMENSIONS'): dims = [int(x) for x in line.split()[1:4]]
        elif line.startswith('SPACING'): spacing = [float(x) for x in line.split()[1:4]]
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
                    vals = [float(x) for x in lines[i].split()]; vecs[j,k] = vals[:3]; i += 1
            data[name] = vecs
        elif line.startswith('SCALARS'):
            name = line.split()[1]; scal = np.zeros((ny, nx)); i += 2
            for j in range(ny):
                for k in range(nx): scal[j,k] = float(lines[i].strip()); i += 1
            data[name] = scal
        else: i += 1
    return data, spacing[0], dims

# ══ Find vortex cores via centerline zero-crossings ══
def find_vortex_cores(vort, x_arr, y_arr, dx, min_strength=0.8, x_start=1.25):
    """Find vortex cores by detecting sign changes of w along the centerline.
    Between each zero-crossing pair, find the peak |w|. This reliably detects
    all alternating vortices in the Karman street."""
    ny, nx = vort.shape
    jc = ny // 2  # centerline index
    x0 = int(x_start/dx)

    # Get centerline vorticity
    w_center = vort[jc, x0:]

    # Find zero-crossings
    signs = np.sign(w_center)
    crossings = []
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1] and signs[i] != 0:
            crossings.append(x0 + i)

    if len(crossings) < 2:
        return []

    # For each segment between crossings, find the peak |w| in 2D
    cores = []
    search_indices = [x0] + crossings + [nx-1]
    for k in range(len(search_indices)-1):
        i_start = search_indices[k]
        i_end   = search_indices[k+1]
        if i_end - i_start < 2:
            continue

        # Find peak |w| in this x-range
        segment = vort[:, i_start:i_end]
        abs_seg = np.abs(segment)
        if abs_seg.max() < min_strength:
            continue

        j_peak, i_peak = np.unravel_index(abs_seg.argmax(), segment.shape)
        i_global = i_start + i_peak

        # Also check a few rows above and below centerline for stronger signal
        best_j, best_i, best_aw = j_peak, i_global, abs_seg[j_peak, i_peak]
        for dj in [-3, -2, -1, 1, 2, 3]:
            jj = jc + dj
            if 0 <= jj < ny:
                local_abs = np.abs(vort[jj, max(i_global-3,0):min(i_global+4,nx)])
                if local_abs.max() > best_aw:
                    best_aw = local_abs.max()
                    li = local_abs.argmax()
                    best_j = jj
                    best_i = max(i_global-3,0) + li

        if best_aw >= min_strength:
            cores.append(dict(x=x_arr[best_i], y=y_arr[best_j],
                              w=vort[best_j, best_i], aw=best_aw))

    # Sort by x position (downstream order)
    cores.sort(key=lambda c: c['x'])
    return cores

# ══ Load data ══
search_dirs = ['/home/liutao/github/UAAMGPCG/output_karman',
               '/home/liutao/github/UAAMGPCG/output_karman_backup']
files = []
for sd in search_dirs:
    if os.path.isdir(sd):
        files = sorted(glob.glob(f'{sd}/frame_*.vtk'))
        if files: print(f"Using {len(files)} VTK files from {sd}"); break
if not files: print("ERROR: No VTK files!"); exit(1)

all_data = []
for fp in files:
    data, dx_v, dims = read_vtk(fp)
    fn = int(re.search(r'frame_(\d+)', fp).group(1))
    t = fn * 0.0078125 * 100
    all_data.append(dict(data=data, dx=dx_v, dims=dims, frame=fn, t=t))
    print(f"  Read {fn:05d} t={t:.2f}s", end='\r')
print(f"\nLoaded {len(all_data)} frames")

nx_pts, ny_pts = all_data[0]['dims'][0], all_data[0]['dims'][1]
dx_val = all_data[0]['dx']
x = np.linspace(0, (nx_pts-1)*dx_val, nx_pts)
y = np.linspace(0, (ny_pts-1)*dx_val, ny_pts)
X, Y = np.meshgrid(x, y)

# ══ Compute scales ══
print("Computing...")
speed_max = 0; asymmetry_hist = []; all_cores = []
for d in all_data:
    vel = d['data']['velocity']; u, v = vel[:,:,0], vel[:,:,1]
    spd = np.sqrt(u**2+v**2); speed_max = max(speed_max, spd.max())
    vort = d['data']['vorticity']
    ccw = vort > 0; cw = vort < 0
    circ_ccw = np.sum(np.abs(vort[ccw]))
    circ_cw  = np.sum(np.abs(vort[cw]))
    tot = circ_ccw + circ_cw
    asym = (circ_ccw - circ_cw)/tot if tot>1e-10 else 0
    asymmetry_hist.append(dict(ccw=circ_ccw, cw=circ_cw, asym=asym,
        ccw_max=np.max(vort[ccw]) if np.any(ccw) else 0,
        cw_max=abs(np.min(vort[cw])) if np.any(cw) else 0))
    all_cores.append(find_vortex_cores(vort, x, y, dx_val))

times = [d['t'] for d in all_data]
VORT_CLIP = 12.0  # clip vorticity colormap to highlight wake

# Mean u for perturbation
u_bulk = np.mean([d['data']['velocity'][:,:,0] for d in all_data], axis=(0,1))

outdir = '/home/liutao/github/UAAMGPCG/diagnostic_output_256'
os.makedirs(outdir, exist_ok=True)

# ═══════════════ ANIMATION ═══════════════
print("Creating animation...")
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(22, 12),
    gridspec_kw={'height_ratios': [1.2, 1], 'width_ratios': [1, 1]})
(ax_flow, ax_vort), (ax_bar, ax_ts) = axes
fig.patch.set_facecolor('#0a0a0f')

skip_s = max(1, nx_pts//60)
xi_s, yi_s = np.meshgrid(np.arange(0, nx_pts, skip_s), np.arange(0, ny_pts, skip_s))

def animate(frame_idx):
    d = all_data[frame_idx]; asym = asymmetry_hist[frame_idx]
    cores = all_cores[frame_idx]
    vel = d['data']['velocity']; u, v = vel[:,:,0], vel[:,:,1]
    speed = np.sqrt(u**2+v**2); vort = d['data']['vorticity']
    solid = d['data'].get('solid', np.zeros_like(speed))
    fn, t_val = d['frame'], d['t']

    for ax in [ax_flow, ax_vort, ax_bar, ax_ts]:
        ax.clear(); ax.set_facecolor('#0a0a0f')

    # ── Panel 1: Flow + perturbed streamlines + vortex markers ──
    ax_flow.pcolormesh(X, Y, speed, shading='auto', cmap='bone', vmin=0, vmax=1.5, rasterized=True)
    u_pert = u - u_bulk
    mask = solid[yi_s, xi_s] < 0.5
    ax_flow.streamplot(x[::skip_s], y[::skip_s],
                        u_pert[yi_s, xi_s]*mask, v[yi_s, xi_s]*mask,
                        color='lime', linewidth=0.5, density=2.5, arrowsize=0.6)
    ax_flow.add_patch(Circle((1.0, 0.5), 0.1, fill=True, facecolor='#333',
                              edgecolor='white', linewidth=3))
    ax_flow.axhline(y=0.5, color='white', linewidth=0.5, linestyle=':', alpha=0.3)

    for c in cores:
        if c['aw'] < 1.5: continue
        is_ccw = c['w'] > 0
        marker = 'o' if is_ccw else 's'
        color = '#ff3333' if is_ccw else '#3388ff'
        ax_flow.scatter(c['x'], c['y'], c=color, s=60+c['aw']*12, zorder=10,
                       marker=marker, edgecolors='white', linewidth=2, alpha=0.95)
        ax_flow.annotate(f'{c["w"]:+.0f}', (c['x'], c['y']),
                        textcoords="offset points", xytext=(0, 14), fontsize=7,
                        color='white', ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7))

    # Legend
    ax_flow.scatter([3.7], [0.94], c='#ff3333', s=100, marker='o', edgecolors='white', linewidth=2)
    ax_flow.text(3.78, 0.94, 'Bottom-shed\n(CCW, +w)', color='#ff4444', fontsize=8, va='center', fontweight='bold')
    ax_flow.scatter([3.7], [0.80], c='#3388ff', s=100, marker='s', edgecolors='white', linewidth=2)
    ax_flow.text(3.78, 0.80, 'Top-shed\n(CW, -w)', color='#4488ff', fontsize=8, va='center', fontweight='bold')

    ax_flow.set_title(f'Flow Field + Vortex Cores    t={t_val:.1f}s    Frame {fn}',
                      fontsize=14, fontweight='bold', color='white')
    ax_flow.set_xlim(0, 4); ax_flow.set_ylim(0, 1)
    ax_flow.set_xlabel('x', color='#aaa'); ax_flow.set_ylabel('y', color='#aaa')
    ax_flow.tick_params(colors='#aaa'); ax_flow.set_aspect(1.0)

    # ── Panel 2: Vorticity + contours ──
    vort_clip = np.clip(vort, -VORT_CLIP, VORT_CLIP)
    ax_vort.pcolormesh(X, Y, vort_clip, shading='auto', cmap='RdBu_r',
                        vmin=-VORT_CLIP, vmax=VORT_CLIP, rasterized=True)
    ax_vort.contour(X, Y, vort, levels=[2,3,4,6,9], colors='#ff3333', linewidths=1.5, alpha=0.8)
    ax_vort.contour(X, Y, vort, levels=[-9,-6,-4,-3,-2], colors='#3388ff',
                    linewidths=1.5, linestyles='--', alpha=0.8)

    for c in cores:
        if c['aw'] < 1.5: continue
        marker = 'o' if c['w'] > 0 else 's'
        ax_vort.scatter(c['x'], c['y'], c='yellow', s=40+c['aw']*6, zorder=10,
                       marker=marker, edgecolors='black', linewidth=1.5)

    ax_vort.add_patch(Circle((1.0, 0.5), 0.1, fill=True, facecolor='#333', edgecolor='white', linewidth=3))

    ax_vort.text(3.6, 0.88, 'BOTTOM-shed\n(+w, CCW)\n== solid contour',
                 color='#ff4444', fontsize=9, ha='center', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, edgecolor='#ff4444'))
    ax_vort.text(3.6, 0.12, 'TOP-shed\n(-w, CW)\n-- dashed contour',
                 color='#4488ff', fontsize=9, ha='center', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, edgecolor='#4488ff'))

    n_ccw = len([c for c in cores if c['w']>0 and c['aw']>1.5])
    n_cw  = len([c for c in cores if c['w']<0 and c['aw']>1.5])
    ax_vort.set_title(f'Vorticity (clipped +-{VORT_CLIP:.0f}) + Contours  |  '
                      f'O Bottom-shed={n_ccw}  [] Top-shed={n_cw}  |  Asym={asym["asym"]*100:+.1f}%',
                      fontsize=13, fontweight='bold', color='white')
    ax_vort.set_xlim(0, 4); ax_vort.set_ylim(0, 1)
    ax_vort.set_xlabel('x', color='#aaa'); ax_vort.set_ylabel('y', color='#aaa')
    ax_vort.tick_params(colors='#aaa'); ax_vort.set_aspect(1.0)

    # ── Panel 3: Circulation comparison ──
    labels = ['Bottom-shed\n(CCW, +w)', 'Top-shed\n(CW, -w)']
    vals = [asym['ccw'], asym['cw']]
    colors_bar = ['#ff3333', '#3388ff']
    bars = ax_bar.bar(labels, vals, color=colors_bar, width=0.5, edgecolor='white', linewidth=2.5)
    for b, val in zip(bars, vals):
        ax_bar.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.02,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=14,
                    fontweight='bold', color='white')
    asym_pct = asym['asym']*100
    stronger = ('Bottom-shed (CCW) STRONGER' if asym_pct > 0.05 else
                ('Top-shed (CW) STRONGER' if asym_pct < -0.05 else 'BALANCED'))
    ax_bar.set_title(f'Circulation  |  Asymmetry = {asym_pct:+.2f}%  ->  {stronger}',
                     fontsize=14, fontweight='bold', color='#ffcc00')
    ax_bar.set_ylabel('Total |w|', color='#aaa')
    ax_bar.tick_params(colors='#aaa'); ax_bar.set_facecolor('#0a0a0f')
    ax_bar.grid(True, alpha=0.15, color='white', axis='y')
    ax_bar.text(0.5, 0.93, f'Peak |w|:  Bottom={asym["ccw_max"]:.1f}  Top={asym["cw_max"]:.1f}',
                transform=ax_bar.transAxes, ha='center', fontsize=10, color='#aaa')

    # ── Panel 4: Time series ──
    ccw_vals = [a['ccw'] for a in asymmetry_hist]
    cw_vals  = [a['cw'] for a in asymmetry_hist]
    ax_ts.plot(times, ccw_vals, '#ff3333', linewidth=2.5, label='Bottom-shed (CCW, +w)')
    ax_ts.plot(times, cw_vals,  '#3388ff', linewidth=2.5, label='Top-shed (CW, -w)')
    ax_ts.fill_between(times, cw_vals, ccw_vals, alpha=0.2, color='#ff3333')
    ax_ts.axvline(x=t_val, color='#ffff00', linewidth=2, alpha=0.7, linestyle='--')
    ax_ts.set_xlabel('Time (s)', color='#aaa')
    ax_ts.set_ylabel('Circulation', color='#aaa')
    ax_ts.set_title('Bottom-shed vs Top-shed Circulation Over Time', fontsize=14, fontweight='bold', color='white')
    ax_ts.legend(fontsize=10, loc='upper left', framealpha=0.7)
    ax_ts.grid(True, alpha=0.15, color='white'); ax_ts.set_facecolor('#0a0a0f')
    ax_ts.tick_params(colors='#aaa')
    ax_ts.scatter([t_val], [ccw_vals[frame_idx]], c='yellow', s=80, zorder=10, edgecolors='black')
    ax_ts.scatter([t_val], [cw_vals[frame_idx]], c='yellow', s=80, zorder=10, edgecolors='black')

    fig.suptitle(f'Karman Vortex Street  |  Re=200, 256x64  |  '
                 f'O Bottom-shed (CCW, +w)  vs  [] Top-shed (CW, -w)  |  Asym={asym_pct:+.1f}%',
                 fontsize=17, fontweight='bold', color='white', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

ani = FuncAnimation(fig, animate, frames=len(all_data), interval=350, blit=False)

gif_path = f'{outdir}/karman_vortex_final.gif'
ani.save(gif_path, writer='pillow', fps=3, dpi=90)
print(f"Saved {gif_path} ({os.path.getsize(gif_path)/1024/1024:.1f} MB)")
plt.close()

# ════ Static snapshot ════
print("Creating static snapshot...")
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(20, 6), facecolor='#0a0a0f')
d_last = all_data[-1]; vort = d_last['data']['vorticity']
vel = d_last['data']['velocity']; speed = np.sqrt(vel[:,:,0]**2+vel[:,:,1]**2)
u_pert = vel[:,:,0] - u_bulk; cores = all_cores[-1]; t_last = times[-1]

a1.pcolormesh(X, Y, speed, shading='auto', cmap='bone', vmin=0, vmax=1.5)
a1.streamplot(x[::skip_s], y[::skip_s], u_pert[yi_s, xi_s], vel[yi_s, xi_s,1],
              color='lime', linewidth=0.6, density=2.5, arrowsize=0.6)
a1.add_patch(Circle((1.0, 0.5), 0.1, fill=True, facecolor='#333', edgecolor='white', linewidth=3))
for c in cores:
    if c['aw'] < 1.5: continue
    is_ccw = c['w'] > 0
    marker = 'o' if is_ccw else 's'
    color = '#ff3333' if is_ccw else '#3388ff'
    a1.scatter(c['x'], c['y'], c=color, s=80+c['aw']*15, zorder=10,
              marker=marker, edgecolors='white', linewidth=2)
a1.set_title(f'Flow + Vortex Cores    t={t_last:.1f}s', fontsize=15, fontweight='bold', color='white')
a1.set_xlim(0,4); a1.set_ylim(0,1); a1.set_aspect(1.0)

vort_clip = np.clip(vort, -VORT_CLIP, VORT_CLIP)
im2 = a2.pcolormesh(X, Y, vort_clip, shading='auto', cmap='RdBu_r', vmin=-VORT_CLIP, vmax=VORT_CLIP)
a2.contour(X, Y, vort, levels=[2,3,4,6,9], colors='#ff3333', linewidths=1.5, alpha=0.8)
a2.contour(X, Y, vort, levels=[-9,-6,-4,-3,-2], colors='#3388ff', linewidths=1.5, linestyles='--', alpha=0.8)
for c in cores:
    if c['aw'] < 1.5: continue
    marker = 'o' if c['w']>0 else 's'
    a2.scatter(c['x'], c['y'], c='yellow', s=60+c['aw']*8, zorder=10,
              marker=marker, edgecolors='black', linewidth=2)
a2.add_patch(Circle((1.0, 0.5), 0.1, fill=True, facecolor='#333', edgecolor='white', linewidth=3))
a2.set_title(f'Vorticity +-{VORT_CLIP:.0f} + Contours    t={t_last:.1f}s', fontsize=15, fontweight='bold', color='white')
a2.set_xlim(0,4); a2.set_ylim(0,1); a2.set_aspect(1.0)
plt.colorbar(im2, ax=a2)

n_ccw = len([c for c in cores if c['w']>0 and c['aw']>1.5])
n_cw  = len([c for c in cores if c['w']<0 and c['aw']>1.5])
last_asym = asymmetry_hist[-1]['asym']*100
fig2.suptitle(f'Karman Vortex Street  |  Re=200  |  O {n_ccw} Bottom-shed (CCW,+w)  +  [] {n_cw} Top-shed (CW,-w)  |  Asym={last_asym:+.1f}%',
             fontsize=16, fontweight='bold', color='white', y=0.98)
plt.tight_layout(rect=[0,0,1,0.95])
fig2.savefig(f'{outdir}/karman_vortex_snapshot.png', dpi=150, facecolor='#0a0a0f', edgecolor='none')
plt.close()
print(f"Saved {outdir}/karman_vortex_snapshot.png")

# ════ Summary ════
asym_pct = [a['asym']*100 for a in asymmetry_hist]
n_ccw_last = len([c for c in all_cores[-1] if c['w']>0 and c['aw']>1.5])
n_cw_last  = len([c for c in all_cores[-1] if c['w']<0 and c['aw']>1.5])

print("\n" + "="*70)
print("  VORTEX VISUALIZATION COMPLETE")
print("="*70)
print(f"  Frames: {len(all_data)}  |  Time: {times[0]:.1f}s - {times[-1]:.1f}s")
print()
print(f"  HOW TO READ THE VISUALIZATION:")
print(f"    Red circles   O = Bottom-shed vortex (CCW, +w)")
print(f"    Blue squares [] = Top-shed vortex    (CW,  -w)")
print(f"    Vortices DRIFT ACROSS the centerline after shedding!")
print(f"    Solid red contours  = +w regions")
print(f"    Dashed blue contours = -w regions")
print(f"    Green streamlines = perturbation velocity (u - U_mean)")
print()
print(f"  FINAL STATE (t={times[-1]:.1f}s):")
print(f"    Bottom-shed (CCW, RED): {n_ccw_last} detected")
print(f"    Top-shed    (CW, BLUE): {n_cw_last} detected")
print(f"    Asymmetry: {asym_pct[-1]:+.2f}%")
print()
print(f"  Output directory: {outdir}/")
print(f"    karman_vortex_final.gif     - Animation")
print(f"    karman_vortex_snapshot.png  - Static snapshot")
print("="*70)
