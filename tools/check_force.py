#!/usr/bin/env python3
"""Check Karman simulation for NaNs and force related quantities."""
import numpy as np, glob, re, subprocess, sys

# Just run a quick diagnostic on the force CSV if it exists
import os
csv_path = '/tmp/karman_validate/force_history.csv'
if os.path.exists(csv_path):
    with open(csv_path) as f:
        lines = f.readlines()
    print(f"Force CSV: {len(lines)-3} data points")
    has_nan = False
    for line in lines[3:]:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            for v in parts:
                try:
                    fv = float(v)
                    if np.isnan(fv) or np.isinf(fv):
                        print(f"  NaN/Inf at t={parts[0]}: Cd={parts[1]} Cl={parts[2]}")
                        has_nan = True
                        break
                except: pass
        if has_nan: break
    if not has_nan:
        print("  No NaN/Inf values found")
else:
    print(f"CSV not found at {csv_path}")
    # Run a quick test to generate it
    print("Running quick force check...")
    result = subprocess.run(['./build/test/test_karman_validate', '64', '1.0'],
                          capture_output=True, text=True, cwd='/home/liutao/github/UAAMGPCG', timeout=120)
    print(result.stdout[-500:])
