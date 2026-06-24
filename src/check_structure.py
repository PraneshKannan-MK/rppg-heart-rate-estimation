"""
check_structure.py
==================
Checks the exact folder structure after extraction.
Run from src\: python check_structure.py
"""
import sys
from pathlib import Path

candidates = [
    Path(r'F:\UST\project_2\rppg_project\data\MR-NIRP-Car'),
    Path(__file__).parent.parent / 'data' / 'MR-NIRP-Car',
]
root = next((c for c in candidates if c.exists()), None)
if not root:
    print("Dataset root not found"); sys.exit(1)

print(f"Root: {root}\n")

# Check first session in detail
for subj in sorted(root.iterdir()):
    if not subj.is_dir(): continue
    for cond in sorted(subj.iterdir()):
        if not cond.is_dir(): continue

        print(f"=== {cond.relative_to(root)} ===")

        # List everything directly inside cond
        for item in sorted(cond.iterdir()):
            if item.is_dir():
                # Count files inside
                files = list(item.iterdir())
                sub_files = [f for f in files if f.is_file()]
                sub_dirs  = [f for f in files if f.is_dir()]
                print(f"  DIR  {item.name}/  ({len(sub_files)} files, {len(sub_dirs)} subdirs)")
                # If it has subdirs, show them too
                for sd in sorted(sub_dirs)[:3]:
                    sd_files = list(sd.glob('*'))
                    print(f"    DIR  {sd.name}/  ({len(sd_files)} files)")
                    # Show first 3 files
                    for f in sorted(sd_files)[:3]:
                        print(f"      {f.name}")
                # Show first 3 files directly in dir
                for f in sorted(sub_files)[:3]:
                    print(f"    FILE {f.name}")
            else:
                size_mb = item.stat().st_size / 1e6
                print(f"  FILE {item.name}  ({size_mb:.1f} MB)")
        print()
        # Only check first 2 sessions
        break
    break