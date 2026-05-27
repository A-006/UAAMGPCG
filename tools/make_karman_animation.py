#!/usr/bin/env python3
"""
Generate Karman vortex street animation from VTK frames in output_karman/.

Reads vorticity and velocity, renders side-by-side panels:
  - Vorticity (RdBu_r, clipped)
  - Velocity magnitude with streamlines

Outputs: output_karman/karman_lfm.mp4 (if ffmpeg) and karman_lfm.gif (always).
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle

# Use the ffmpeg bundled with imageio-ffmpeg (no system install needed).
try:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass


def read_vtk(path):
    """Parse the structured-points VTK files written by VtkWriter."""
    with open(path) as f:
        lines = f.readlines()
    dims = spacing = None
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('DIMENSIONS'):
            dims = [int(x) for x in s.split()[1:4]]
        elif s.startswith('SPACING'):
            spacing = [float(x) for x in s.split()[1:4]]
        elif s.startswith('POINT_DATA'):
            break
        i += 1
    nx, ny, _ = dims
    data = {}
    i += 1
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith('VECTORS'):
            name = s.split()[1]
            arr = np.zeros((ny, nx, 3), dtype=np.float32)
            i += 1
            for jj in range(ny):
                for ii in range(nx):
                    arr[jj, ii] = [float(v) for v in lines[i].split()[:3]]
                    i += 1
            data[name] = arr
        elif s.startswith('SCALARS'):
            name = s.split()[1]
            arr = np.zeros((ny, nx), dtype=np.float32)
            i += 2
            for jj in range(ny):
                for ii in range(nx):
                    arr[jj, ii] = float(lines[i].strip())
                    i += 1
            data[name] = arr
        else:
            i += 1
    return data, spacing[0], spacing[1], dims


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'output_karman'
    files = sorted(glob.glob(os.path.join(out_dir, 'frame_*.vtk')))
    if not files:
        print(f"No frames in {out_dir}")
        return 1
    print(f"Found {len(files)} frames in {out_dir}")

    frames = []
    for fp in files:
        d, dx, dy, dims = read_vtk(fp)
        m = re.search(r'frame_(\d+)', fp)
        frame_num = int(m.group(1)) if m else 0
        frames.append({
            'data': d, 'dx': dx, 'dy': dy, 'dims': dims, 'frame': frame_num
        })
    print(f"Loaded {len(frames)} frames")

    nx_pts, ny_pts = frames[0]['dims'][0], frames[0]['dims'][1]
    dx = frames[0]['dx']
    dy = frames[0]['dy']
    x = np.linspace(0, (nx_pts - 1) * dx, nx_pts)
    y = np.linspace(0, (ny_pts - 1) * dy, ny_pts)
    X, Y = np.meshgrid(x, y)
    Lx = (nx_pts - 1) * dx
    Ly = (ny_pts - 1) * dy

    vort_max = max(abs(f['data']['vorticity']).max() for f in frames)
    speed_max = max(np.sqrt(f['data']['velocity'][..., 0] ** 2
                            + f['data']['velocity'][..., 1] ** 2).max()
                    for f in frames)
    vort_clip = max(min(vort_max * 0.4, 30.0), 5.0)
    print(f"vorticity peak={vort_max:.2f}, clipped to +-{vort_clip:.1f}")
    print(f"speed max={speed_max:.2f}")

    plt.style.use('dark_background')
    fig, (ax_v, ax_s) = plt.subplots(2, 1, figsize=(13, 6.2), facecolor='#0a0a0f')
    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.06, hspace=0.18)

    skip = max(1, nx_pts // 70)
    xi = np.arange(0, nx_pts, skip)
    yi = np.arange(0, ny_pts, skip)

    cyl_cx, cyl_cy, cyl_R = 1.0, 0.5, 0.1

    def render(idx):
        f = frames[idx]
        vort = f['data']['vorticity']
        vel = f['data']['velocity']
        u = vel[..., 0]
        v = vel[..., 1]
        speed = np.sqrt(u * u + v * v)
        solid = f['data'].get('solid', np.zeros_like(speed))

        for ax in (ax_v, ax_s):
            ax.clear()
            ax.set_facecolor('#0a0a0f')

        ax_v.pcolormesh(X, Y, np.clip(vort, -vort_clip, vort_clip),
                        cmap='RdBu_r', vmin=-vort_clip, vmax=vort_clip,
                        shading='auto', rasterized=True)
        ax_v.add_patch(Circle((cyl_cx, cyl_cy), cyl_R, fill=True,
                              facecolor='#222', edgecolor='white', linewidth=1.5))
        ax_v.set_xlim(0, Lx)
        ax_v.set_ylim(0, Ly)
        ax_v.set_aspect('equal')
        ax_v.set_title(f'Vorticity   frame {f["frame"]}',
                       color='white', fontsize=11)
        ax_v.tick_params(colors='#aaa', labelsize=8)

        ax_s.pcolormesh(X, Y, speed, cmap='magma', vmin=0,
                        vmax=min(speed_max, 1.8 * 1.0),
                        shading='auto', rasterized=True)
        mask = (solid[np.ix_(yi, xi)] < 0.5).astype(float)
        ax_s.streamplot(x[xi], y[yi],
                        u[np.ix_(yi, xi)] * mask,
                        v[np.ix_(yi, xi)] * mask,
                        color='lime', linewidth=0.45, density=2.0,
                        arrowsize=0.5)
        ax_s.add_patch(Circle((cyl_cx, cyl_cy), cyl_R, fill=True,
                              facecolor='#222', edgecolor='white', linewidth=1.5))
        ax_s.set_xlim(0, Lx)
        ax_s.set_ylim(0, Ly)
        ax_s.set_aspect('equal')
        ax_s.set_title('Speed + streamlines', color='white', fontsize=11)
        ax_s.tick_params(colors='#aaa', labelsize=8)

        fig.suptitle(f'LFM Karman vortex street  |  '
                     f'grid {nx_pts-1}x{ny_pts-1}  |  Re=200  |  '
                     f'frame {idx+1}/{len(frames)}',
                     color='white', fontsize=13, y=0.985)

    ani = FuncAnimation(fig, render, frames=len(frames), interval=80, blit=False)

    # MP4 (H.264 + yuv420p) — phone-friendly, plays inline in WeChat.
    try:
        mp4_path = os.path.join(out_dir, 'karman_lfm.mp4')
        print(f"Writing {mp4_path} ...")
        writer = FFMpegWriter(fps=20, bitrate=4000,
                              codec='libx264',
                              extra_args=['-pix_fmt', 'yuv420p'])
        ani.save(mp4_path, writer=writer, dpi=110)
        print(f"  size: {os.path.getsize(mp4_path)/1024/1024:.2f} MB")
    except Exception as e:
        print(f"  mp4 skipped: {e}")

    # GIF — fallback / preview.
    gif_path = os.path.join(out_dir, 'karman_lfm.gif')
    print(f"Writing {gif_path} ...")
    ani.save(gif_path, writer=PillowWriter(fps=12), dpi=80)
    print(f"  size: {os.path.getsize(gif_path)/1024/1024:.2f} MB")

    plt.close(fig)
    return 0


if __name__ == '__main__':
    sys.exit(main())
