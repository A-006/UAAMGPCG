#!/usr/bin/env python3
"""Quick numerical analysis of VTK output for Karman diagnostics."""
import numpy as np
import re, glob, sys

def read_vtk_ascii(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    dims = None; origin = [0,0,0]; spacing = [1,1,1]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('DIMENSIONS'): dims = [int(x) for x in line.split()[1:4]]
        elif line.startswith('ORIGIN'): origin = [float(x) for x in line.split()[1:4]]
        elif line.startswith('SPACING'): spacing = [float(x) for x in line.split()[1:4]]
        elif line.startswith('POINT_DATA'): break
        i += 1
    nx, ny, nz = dims; npts = nx * ny
    data = {}; i += 1
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
    return data, origin, spacing, (nx, ny, nz)

files = sorted(glob.glob('/home/liutao/github/UAAMGPCG/output_karman/frame_*.vtk'))
fp = files[-1]
data, origin, spacing, dims = read_vtk_ascii(fp)
vel = data['velocity']; u, v = vel[:,:,0], vel[:,:,1]
vort = data['vorticity']; divg = data['divergence']; solid = data['solid']
dx = spacing[0]; ny, nx = u.shape; jc = ny // 2

print('=== Frame 31 Final State ===')
print(f'Grid: {nx}x{ny}, dx={dx:.4f}, D/dx={0.2/dx:.1f} cells/D')

print(f'\n--- Centerline velocity ---')
for xi in range(int(0.8/dx), int(3.5/dx), int(0.2/dx)):
    x = xi * dx
    print(f'  x={x:.3f}  u={u[jc,xi]:.4f}  v={v[jc,xi]:.4f}  vort={vort[jc,xi]:.3f}')

print(f'\n--- v-velocity cross-sections ---')
for x_check in [1.5, 2.0, 2.5, 3.0]:
    xi = int(x_check/dx)
    vs = v[:, xi]
    print(f'  x={x_check:.1f}: v=[{vs.min():.4f}, {vs.max():.4f}]')

print(f'\n--- Vorticity in wake (x>1.2) ---')
wm = (np.arange(nx)*dx > 1.2) & (solid[ny//2,:] < 0.5)
wv = vort[:, wm]
print(f'  pos(>1): {(wv>1).sum()}  neg(<-1): {(wv<-1).sum()}  max|w|={abs(wv).max():.1f}')

print(f'\n--- Vorticity at x=2.0 ---')
xi = int(2.0/dx)
for j in range(0, ny, 4):
    print(f'  y={j*dx:.3f}  vort={vort[j,xi]:.3f}')

# Check if flow is at all unsteady
print(f'\n--- Time evolution (every 4th frame) ---')
for fp in files[::4]:
    data, _, _, _ = read_vtk_ascii(fp)
    vel = data['velocity']; vmid = vel[ny//2,:,1]
    vort = data['vorticity']
    match = re.search(r'frame_(\d+)', fp)
    fn = int(match.group(1))
    # v at x=2.0 centerline
    xi = int(2.0/dx)
    v_cl = vmid[xi] if xi < len(vmid) else 0
    print(f'  Frame {fn}: v(x=2,y=0.5)={v_cl:.4f}  max|vort|={abs(vort).max():.1f}')
