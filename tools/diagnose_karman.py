#!/usr/bin/env python3
"""
Karman Vortex Street Diagnostic Tool
Analyzes VTK output to identify why vortex shedding fails.
"""
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os, sys, glob, re

def read_vtk_ascii(filepath):
    """Parse STRUCTURED_POINTS ASCII VTK file."""
    with open(filepath) as f:
        lines = f.readlines()

    # Parse header
    dims = None
    origin = [0,0,0]
    spacing = [1,1,1]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('DIMENSIONS'):
            dims = [int(x) for x in line.split()[1:4]]
        elif line.startswith('ORIGIN'):
            origin = [float(x) for x in line.split()[1:4]]
        elif line.startswith('SPACING'):
            spacing = [float(x) for x in line.split()[1:4]]
        elif line.startswith('POINT_DATA'):
            break
        i += 1

    nx, ny, nz = dims
    npts = nx * ny
    data = {}
    i += 1

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
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
            i += 2  # skip LOOKUP_TABLE
            for j in range(ny):
                for k in range(nx):
                    scal[j, k] = float(lines[i].strip())
                    i += 1
            data[name] = scal
        else:
            i += 1

    return data, origin, spacing, (nx, ny, nz)

def compute_vorticity_checks(data, dx):
    """Compute vorticity and check if it's non-zero."""
    vel = data['velocity']
    u = vel[:,:,0]
    v = vel[:,:,1]
    ny, nx = u.shape
    omega = np.zeros((ny, nx))
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            dvdx = (v[j, i+1] - v[j, i-1]) / (2*dx)
            dudy = (u[j+1, i] - u[j-1, i]) / (2*dx)
            omega[j, i] = dvdx - dudy
    return omega

def plot_diagnostic_panel(data, origin, spacing, dims, frame_idx, time_val, outdir):
    """Create comprehensive diagnostic plot for one frame."""
    nx_pts, ny_pts = dims[0], dims[1]
    dx, dy = spacing[0], spacing[1]
    x = np.linspace(origin[0], origin[0] + (nx_pts-1)*dx, nx_pts)
    y = np.linspace(origin[1], origin[1] + (ny_pts-1)*dy, ny_pts)
    X, Y = np.meshgrid(x, y)

    vel = data['velocity']
    u = vel[:,:,0]
    v = vel[:,:,1]
    speed = np.sqrt(u**2 + v**2)
    vort = data.get('vorticity', np.zeros_like(speed))
    divg = data.get('divergence', np.zeros_like(speed))
    solid = data.get('solid', np.zeros_like(speed))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f'Karman Diagnostic — Frame {frame_idx}  t={time_val:.3f}s', fontsize=14, fontweight='bold')

    # 1) Velocity magnitude + streamlines
    ax = axes[0,0]
    im = ax.pcolormesh(X, Y, speed, shading='auto', cmap='jet')
    # Subsample for streamlines
    skip = max(1, nx_pts // 80)
    xi, yi = np.meshgrid(np.arange(0, nx_pts, skip), np.arange(0, ny_pts, skip))
    us = u[yi, xi]
    vs = v[yi, xi]
    # Mask solid regions in streamline source
    mask = solid[yi, xi] < 0.5
    ax.streamplot(x[::skip], y[::skip], us*mask, vs*mask, color='white', linewidth=0.6, density=1.5, arrowsize=0.6)
    ax.set_title(f'Velocity |u| (max={speed.max():.3f})')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(0, 4); ax.set_ylim(0, 1)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
    plt.colorbar(im, ax=ax)

    # 2) Vorticity field
    ax = axes[0,1]
    vmax = max(abs(vort.min()), abs(vort.max()), 0.01)
    im = ax.pcolormesh(X, Y, vort, shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'Vorticity ω = dv/dx - du/dy (max|ω|={vmax:.3f})')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(0, 4); ax.set_ylim(0, 1)
    plt.colorbar(im, ax=ax)

    # 3) Divergence
    ax = axes[0,2]
    dmax = max(abs(divg.min()), abs(divg.max()), 0.001)
    im = ax.pcolormesh(X, Y, divg, shading='auto', cmap='RdBu_r', vmin=-dmax, vmax=dmax)
    ax.set_title(f'Divergence ∇·u (max|∇·u|={dmax:.4f})')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(0, 4); ax.set_ylim(0, 1)
    plt.colorbar(im, ax=ax)

    # 4) Solid mask + cylinder
    ax = axes[1,0]
    im = ax.pcolormesh(X, Y, solid, shading='auto', cmap='gray', vmin=0, vmax=1)
    cylinder = Circle((1.0, 0.5), 0.1, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(cylinder)
    ax.set_title('Solid Mask (black=solid) + Reference Cylinder')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(0.5, 1.8); ax.set_ylim(0.15, 0.85)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # 5) Velocity profile at y=0.5 (centerline)
    ax = axes[1,1]
    jc = ny_pts // 2
    ax.plot(x, u[jc,:], 'b-', label='u (x-vel)', linewidth=1)
    ax.plot(x, v[jc,:], 'r-', label='v (y-vel)', linewidth=1)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='cylinder cx')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_title('Velocity along centerline y=0.5')
    ax.set_xlabel('x')
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, 3.0)

    # 6) Divergence along y=0.5
    ax = axes[1,2]
    ax.plot(x, divg[jc,:], 'g-', linewidth=1)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Divergence along centerline y=0.5')
    ax.set_xlabel('x')
    ax.set_xlim(0.5, 3.0)

    plt.tight_layout()
    fname = f'{outdir}/diag_frame_{frame_idx:05d}.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    return fname

def plot_comparison_timeline(vtk_files, outdir):
    """Compare early, middle, late frames side by side."""
    if len(vtk_files) < 3:
        return

    indices = [0, len(vtk_files)//3, 2*len(vtk_files)//3, len(vtk_files)-1]
    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    fig.suptitle('Karman Vortex Street — Time Evolution', fontsize=14, fontweight='bold')

    for row, idx in enumerate(indices):
        fp = vtk_files[idx]
        data, origin, spacing, dims = read_vtk_ascii(fp)
        nx_pts, ny_pts = dims[0], dims[1]
        dx = spacing[0]
        x = np.linspace(origin[0], origin[0] + (nx_pts-1)*dx, nx_pts)
        y = np.linspace(origin[1], origin[1] + (ny_pts-1)*dx, ny_pts)
        X, Y = np.meshgrid(x, y)

        vel = data['velocity']
        speed = np.sqrt(vel[:,:,0]**2 + vel[:,:,1]**2)
        vort = data.get('vorticity', np.zeros_like(speed))
        divg = data.get('divergence', np.zeros_like(speed))

        # Extract frame number and approximate time
        match = re.search(r'frame_(\d+)', fp)
        fn = int(match.group(1)) if match else idx
        t_approx = fn * 0.015625 * 100  # dt * frame_skip * frame

        # Speed
        ax = axes[row, 0]
        im = ax.pcolormesh(X, Y, speed, shading='auto', cmap='jet')
        ax.set_title(f'Frame {fn} — Speed |u| (max={speed.max():.2f})')
        ax.set_xlim(0, 4); ax.set_ylim(0, 1)
        plt.colorbar(im, ax=ax)

        # Vorticity
        ax = axes[row, 1]
        vm = max(abs(vort.min()), abs(vort.max()), 0.01)
        im = ax.pcolormesh(X, Y, vort, shading='auto', cmap='RdBu_r', vmin=-vm, vmax=vm)
        ax.set_title(f'Vorticity (max|ω|={vm:.2f})')
        ax.set_xlim(0, 4); ax.set_ylim(0, 1)
        plt.colorbar(im, ax=ax)

        # Divergence
        ax = axes[row, 2]
        dm = max(abs(divg.min()), abs(divg.max()), 0.001)
        im = ax.pcolormesh(X, Y, divg, shading='auto', cmap='RdBu_r', vmin=-dm, vmax=dm)
        ax.set_title(f'Divergence (max|∇·u|={dm:.3f})')
        ax.set_xlim(0, 4); ax.set_ylim(0, 1)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fname = f'{outdir}/comparison_timeline.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname

def plot_divergence_growth(vtk_files, outdir):
    """Plot max divergence over time to show accumulation."""
    max_divs = []
    frame_nums = []
    for fp in vtk_files:
        data, _, _, _ = read_vtk_ascii(fp)
        divg = data.get('divergence', np.zeros(1))
        max_divs.append(np.abs(divg).max())
        match = re.search(r'frame_(\d+)', fp)
        frame_nums.append(int(match.group(1)) if match else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame_nums, max_divs, 'r-o', markersize=3)
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Max |divergence|')
    ax.set_title('Divergence Accumulation Over Time')
    ax.axhline(y=1e-3, color='gray', linestyle='--', alpha=0.5, label='Target: 1e-3')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fname = f'{outdir}/divergence_growth.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    return fname

def print_summary(vtk_files):
    """Print text summary of diagnostics."""
    first = read_vtk_ascii(vtk_files[0])
    last = read_vtk_ascii(vtk_files[-1])

    print("\n" + "="*60)
    print("  DIAGNOSTIC SUMMARY")
    print("="*60)

    for label, data in [("First frame", first), ("Last frame", last)]:
        vel = data[0]['velocity']
        speed = np.sqrt(vel[:,:,0]**2 + vel[:,:,1]**2)
        vort = data[0].get('vorticity', np.zeros_like(speed))
        divg = data[0].get('divergence', np.zeros_like(speed))
        solid = data[0].get('solid', np.zeros_like(speed))

        nsolid = int(solid.sum())
        max_vort = abs(vort).max()

        print(f"\n  {label}:")
        print(f"    Max speed:      {speed.max():.3f}")
        print(f"    Max |vorticity|: {max_vort:.4f}")
        print(f"    Max |div|:       {abs(divg).max():.4f}")
        print(f"    Solid cells:     {nsolid}")
        print(f"    Vortex present:  {'YES' if max_vort > 0.5 else 'NO (too weak)'}")

    # Check solid mask
    solid = last[0]['solid']
    ny, nx = solid.shape
    # Find solid cell range
    solid_x = []
    for j in range(ny):
        for i in range(nx):
            if solid[j,i] > 0.5:
                solid_x.append(i)
    if solid_x:
        print(f"\n  Solid cells x-range (pixels): {min(solid_x)}-{max(solid_x)}")
        print(f"  Expected cylinder: x=1.0, R=0.1 → pixel {1.0/0.03125:.0f}")

    print("\n" + "="*60)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Karman Vortex Diagnostic Tool')
    parser.add_argument('vtk_dir', nargs='?', default='output_karman',
                        help='Directory containing frame_*.vtk files')
    parser.add_argument('--out', default='diagnostic_output',
                        help='Output directory for diagnostic plots')
    parser.add_argument('--frames', type=int, default=0,
                        help='Number of frames to analyze (0=all)')
    parser.add_argument('--skip', type=int, default=1,
                        help='Process every Nth frame')
    args = parser.parse_args()

    vtk_files = sorted(glob.glob(f'{args.vtk_dir}/frame_*.vtk'))
    if not vtk_files:
        print(f"ERROR: No VTK files found in {args.vtk_dir}")
        sys.exit(1)

    if args.frames > 0:
        step = max(1, len(vtk_files) // args.frames)
        vtk_files = vtk_files[::step]
    elif args.skip > 1:
        vtk_files = vtk_files[::args.skip]

    print(f"Found {len(vtk_files)} VTK files to analyze")
    os.makedirs(args.out, exist_ok=True)

    # Print text summary
    print_summary(vtk_files)

    # Generate comparison timeline
    print("\nGenerating comparison timeline...")
    plot_comparison_timeline(vtk_files, args.out)
    print(f"  -> {args.out}/comparison_timeline.png")

    # Divergence growth
    print("Generating divergence growth plot...")
    plot_divergence_growth(vtk_files, args.out)
    print(f"  -> {args.out}/divergence_growth.png")

    # Detailed panels for key frames
    key_indices = [0, len(vtk_files)//4, len(vtk_files)//2, 3*len(vtk_files)//4, len(vtk_files)-1]
    for idx in key_indices:
        if idx >= len(vtk_files):
            continue
        fp = vtk_files[idx]
        match = re.search(r'frame_(\d+)', fp)
        fn = int(match.group(1)) if match else idx
        print(f"Generating diagnostic panel for frame {fn}...")
        data, origin, spacing, dims = read_vtk_ascii(fp)
        plot_diagnostic_panel(data, origin, spacing, dims, fn, fn*0.015625*100, args.out)

    print(f"\nDone! All plots saved to {args.out}/")
    print("Key files:")
    print(f"  {args.out}/comparison_timeline.png  — time evolution overview")
    print(f"  {args.out}/divergence_growth.png     — divergence accumulation")
    print(f"  {args.out}/diag_frame_*.png          — per-frame detailed diagnostics")

if __name__ == '__main__':
    main()
