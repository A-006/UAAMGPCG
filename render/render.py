#!/usr/bin/env python3
"""Unified post-processing entry — reads the SAME .in input file as cfdsim and
renders that run's output with the appropriate renderer.

It parses the input file exactly like cfdsim (INI `key = value`, `#` comments,
trailing `key=value` CLI args override the file), pulls out `scenario` / `out_dir`
and the `render_*` knobs, then dispatches to the renderers in this directory.

Usage:
  python3 render/render.py <input.in> [key=value]...
  python3 render/render.py inputs/vortex_ring.in
  python3 render/render.py inputs/collision_paper.in out_dir=output_conv160 render_mode=slice

Render knobs (put them in the .in file, or override on the CLI):
  render_mode    comma list of: iso3d | slice | volume | none   (default iso3d)
  render_level   iso-level as a fraction of p99.5 |omega|        (default 0.15)
  render_fps     frames per second                               (default 12)
  render_orbit   turntable azimuth sweep, deg (iso3d only)       (default 120)
  render_cmap    colormap                                        (default turbo)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Default out_dir per scenario — mirrors cfdsim's apply_scene_presets so the
# renderer finds the frames even when the .in file omits out_dir.
SCENE_OUT = {
    "vortex_ring": "output_vortex_ring",
    "vortex_collision": "output_vortex_collision",
    "collision_paper": "output_collision_paper",
    "delta_wing": "output_delta_wing",
    "vortex_reconnection": "output_vortex_reconnection",
    "trefoil_knot": "output_trefoil",
}


def parse(argv):
    """file values then CLI key=value (CLI wins), same precedence as cfdsim."""
    kv = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k.strip()] = v.strip()
        else:  # bare arg = input file path
            with open(a) as f:
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    if "=" in line:
                        k, v = line.split("=", 1)
                        kv[k.strip()] = v.strip()
    return kv


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    kv    = parse(sys.argv[1:])
    scen  = kv.get("scenario", "vortex_ring")
    out   = kv.get("out_dir", SCENE_OUT.get(scen, "output"))
    modes = [m.strip() for m in kv.get("render_mode", "iso3d").split(",") if m.strip()]
    level = kv.get("render_level", "0.15")
    fps   = kv.get("render_fps", "12")
    orbit = kv.get("render_orbit", "120")
    cmap  = kv.get("render_cmap", "turbo")
    # head-on collisions expand in the y-z plane → use the collision slice layout
    slice_mode = "collision_x" if scen in ("vortex_collision", "collision_paper") else "side_z"

    if not os.path.isdir(out):
        sys.exit(f"render: output dir '{out}' not found — run the sim first "
                 f"(./build/cfdsim ... out_dir={out})")

    print(f"render: scenario={scen}  out={out}/  modes={modes}")
    for m in modes:
        if m == "none":
            continue
        if m == "iso3d":
            cmd = [sys.executable, f"{HERE}/render3d_pv.py", out,
                   f"{out}/{scen}_3d.gif", level, fps, orbit, cmap]
        elif m == "slice":
            cmd = [sys.executable, f"{HERE}/render_vortex_ring.py", out,
                   f"{out}/{scen}_slice.gif", fps, slice_mode]
        elif m == "volume":
            cmd = [sys.executable, f"{HERE}/render_volume.py", out,
                   f"{out}/{scen}_vol.gif"]
        else:
            print(f"render: unknown render_mode '{m}' (skipped)")
            continue
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print("render: done")


if __name__ == "__main__":
    main()
