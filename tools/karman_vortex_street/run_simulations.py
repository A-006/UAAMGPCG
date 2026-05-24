"""
Run Karman vortex street simulations for all solvers and collect timing/accuracy data.
"""

import os, sys, subprocess, time, re, json
from config import *
from generate_inputs import generate_all

def run_one(input_path: str) -> dict:
    """Run a single simulation and return metrics."""
    label = os.path.basename(input_path).replace(".txt", "")
    print(f"  Running {label} ... ", end="", flush=True)

    if not os.path.exists(LFM_EXE):
        print(f"SKIP (lfm_2d not found at {LFM_EXE})")
        return {"label": label, "error": "executable not found"}

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [LFM_EXE, input_path],
            capture_output=True, text=True, timeout=600,
            cwd=PROJECT_ROOT
        )
        elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"label": label, "error": "timeout"}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"label": label, "error": str(e)}

    output = result.stdout + result.stderr

    # Parse wall-clock time
    time_match = re.search(r"(\d+)\s+steps\s+in\s+([\d.]+)\s+s", output)
    nsteps = int(time_match.group(1)) if time_match else 0
    wall_time = float(time_match.group(2)) if time_match else 0.0

    # Parse per-step time
    ms_match = re.search(r"\(([\d.]+)\s+ms/step\)", output)
    ms_per_step = float(ms_match.group(1)) if ms_match else 0.0

    # Parse final max divergence (from last VtkWriter::printStatus)
    div_matches = re.findall(r"max\|div\|=([\d.e+\-]+)", output)
    final_div = float(div_matches[-1]) if div_matches else float("nan")

    # Parse final max velocity
    vel_matches = re.findall(r"max\|u\|=([\d.]+)", output)
    final_vel = float(vel_matches[-1]) if vel_matches else float("nan")

    print(f"{elapsed:.1f}s wall, {ms_per_step:.1f} ms/step, max|div|={final_div:.2e}")

    return {
        "label": label,
        "wall_time": wall_time,
        "ms_per_step": ms_per_step,
        "nsteps": nsteps,
        "final_div": final_div,
        "final_vel": final_vel,
        "elapsed_real": elapsed,
    }

def run_all(params: KarmanParams = KarmanParams()):
    """Run simulations for all solver/grid combinations."""
    # Generate inputs first
    print("Generating INPUT files ...")
    input_files = generate_all(params)

    results = []
    print(f"\nRunning simulations ({LFM_EXE}) ...")
    for (solver_key, grid_label), input_path in sorted(input_files.items()):
        r = run_one(input_path)
        r["solver"] = solver_key
        r["grid"] = grid_label
        results.append(r)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results

if __name__ == "__main__":
    run_all()
