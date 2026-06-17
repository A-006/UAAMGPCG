#!/usr/bin/env python3
"""Real 3D iso-surface rendering of LFM vorticity frames, via PyVista (VTK).

Unlike the matplotlib scripts (slices / projections / flat marching-cubes), this
renders a properly lit, depth-occluded 3D iso-surface of |omega| — what you want
to actually SEE the donut-shaped ring and (for collisions) the radial filaments.

The iso-surface is coloured by VELOCITY MAGNITUDE when the frames carry a
velocity field (vtk_mode=full), otherwise a solid colour — note |omega| itself is
~constant on an |omega| iso-surface, so colouring by it would be flat/dark.

Runs head-less: VTK falls back to EGL/OSMesa off-screen rendering, so no display
is needed (you may see a harmless "bad X server connection" warning).

Usage:
  render3d_pv.py <frames_dir> [out.gif] [level_frac] [fps] [orbit_deg] [cmap]
    frames_dir   directory of frame_*.vtk
    out.gif      output animation              (default <dir>/render3d.gif)
    level_frac   iso-level as a fraction of the run's 99.5th-pct |omega|
                                               (default 0.15; smaller = fatter)
    fps          frames per second             (default 12)
    orbit_deg    total camera azimuth swept over the run, turntable feel
                                               (default 0 = fixed view)
    cmap         matplotlib colormap name      (default turbo)

  e.g.  python3 tools/render3d_pv.py output_vortex_ring ring3d.gif 0.15 12 90
"""
import glob
import os
import sys

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

BG = (0.08, 0.09, 0.12)  # dark blue-grey: shows the lit surface better than pure black


def umag(mesh):
    return np.linalg.norm(mesh["velocity"], axis=1) if "velocity" in mesh.point_data else None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    frames_dir = sys.argv[1]
    out        = sys.argv[2] if len(sys.argv) > 2 else os.path.join(frames_dir, "render3d.gif")
    level_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    fps        = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    orbit_deg  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    cmap       = sys.argv[6] if len(sys.argv) > 6 else "turbo"

    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.vtk")))
    if not files:
        sys.exit(f"render3d_pv: no frame_*.vtk in {frames_dir}")

    # Global scales (so the surface + colours don't flicker frame-to-frame):
    #   vmax  — 99.5th-pct |omega| for the iso-level
    #   clim  — velocity-magnitude range on the iso-surface (if velocity present)
    sample = files[:: max(1, len(files) // 8)]
    vmax = 0.0
    for f in sample:
        vmax = max(vmax, float(np.percentile(pv.read(f)["vorticity_magnitude"], 99.5)))
    iso = level_frac * vmax

    lo, hi, color_by_vel = 1e30, -1e30, False
    for f in sample:
        g = pv.read(f)
        g.set_active_scalars("vorticity_magnitude")
        s = g.contour([iso])
        m = umag(s)
        if m is not None and m.size:
            color_by_vel = True
            lo, hi = min(lo, float(m.min())), max(hi, float(m.max()))
    print(f"render3d_pv: {len(files)} frames  iso={iso:.3f} (= {level_frac} x p99.5={vmax:.3f})  "
          f"color={'velocity' if color_by_vel else 'solid'}  cmap={cmap}  orbit={orbit_deg} deg")

    p = pv.Plotter(off_screen=True, window_size=(960, 760))
    p.set_background(BG)
    p.enable_lightkit()  # 3-point studio lighting → strong sense of depth
    p.open_gif(out, fps=fps)

    n     = len(files)
    delta = (orbit_deg / (n - 1)) if (orbit_deg and n > 1) else 0.0
    cam_set = False
    for i, f in enumerate(files):
        g = pv.read(f)
        g.set_active_scalars("vorticity_magnitude")
        surf = g.contour([iso])
        p.clear()
        if surf.n_points:
            if color_by_vel:
                surf["umag"] = umag(surf)
                p.add_mesh(surf, scalars="umag", cmap=cmap, clim=(lo, hi),
                           smooth_shading=True, specular=0.4, specular_power=20,
                           show_scalar_bar=False, reset_camera=not cam_set)
            else:
                p.add_mesh(surf, color="darkorange", smooth_shading=True, specular=0.4,
                           specular_power=20, show_scalar_bar=False, reset_camera=not cam_set)
        if not cam_set:
            p.camera_position = "iso"
            p.camera.zoom(1.4)
            cam_set = True
        if delta:
            p.camera.Azimuth(delta)  # turntable sweep
        p.add_text(f"frame {i:>4d}", color="white", font_size=9)
        p.write_frame()
    p.close()
    print("render3d_pv: wrote", out)


if __name__ == "__main__":
    main()
