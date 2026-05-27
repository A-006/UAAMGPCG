#!/usr/bin/env python3
"""Check for vortex shedding evidence in 256 resolution simulation."""
import numpy as np, glob, re

def read_vtk(filepath):
    with open(filepath) as f: lines = f.readlines()
    dims = None; spacing = [1,1,1]
    i = 0
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
                for k in range(nx): vals = [float(x) for x in lines[i].split()]; vecs[j,k] = vals[:3]; i += 1
            data[name] = vecs
        elif line.startswith('SCALARS'):
            name = line.split()[1]; scal = np.zeros((ny, nx)); i += 2
            for j in range(ny):
                for k in range(nx): scal[j,k] = float(lines[i].strip()); i += 1
            data[name] = scal
        else: i += 1
    return data, spacing[0]

files = sorted(glob.glob('/home/liutao/github/UAAMGPCG/output_karman/frame_*.vtk'))

# Track v at monitoring point (x=2.0, y=0.5) over time
print('Time evolution of v(x=2.0, y=0.5):')
print(f'{"Frame":>8s} {"t_approx":>10s} {"v(2,0.5)":>12s}')
v_history = []
for fp in files:
    data, dx = read_vtk(fp)
    vel = data['velocity']; v = vel[:,:,1]
    ny, nx = v.shape
    jc = ny // 2
    xi = int(2.0/dx)
    v_val = v[jc, xi] if xi < nx else 0
    match = re.search(r'frame_(\d+)', fp)
    fn = int(match.group(1))
    t_approx = fn * 0.0078125 * 100  # dt * frame_skip
    v_history.append(v_val)
    print(f'{fn:8d} {t_approx:10.4f} {v_val:12.6f}')

# Check for sign changes (oscillation)
v_arr = np.array(v_history)
sign_changes = np.sum(np.diff(np.signbit(v_arr)))
print(f'\nSign changes in v: {sign_changes}')
print(f'v range: [{v_arr.min():.4f}, {v_arr.max():.4f}]')
print(f'v amplitude: {(v_arr.max()-v_arr.min())/2:.4f}')

if sign_changes > 0:
    print('\n*** VORTEX SHEDDING DETECTED! v-velocity oscillates ***')
else:
    print('\nNo vortex shedding detected (v monotonic)')
