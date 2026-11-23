#!/usr/bin/env python3
r"""
Step 08.2: repeat Step 08 after excluding the donor that dominates the
resolution-0.3 state3 solution.

This is intentionally a thin launcher around
``08_combine_recluster_final_LEC_age_quartiles.py``. Consequently, HVG
selection, scaling, PCA, Harmony, neighbors, UMAP, Leiden resolution, marker
ranking, summaries, and figures are produced by the exact same implementation
as Step 08. The only analytical change is removal of
``GSE183852::TWCM-13-80`` before HVG selection and clustering.

Default output
--------------
new_result2_aging/08_2_combined_LEC_recluster_no_dominant_donor/dataset13/

The source Step 08 inputs and outputs are not modified. All Step 08.2 output
filenames begin with ``08_2_``.

Example
-------
    .\.venv\Scripts\python.exe `
      .\new_script2_aging\08_2_recluster_LECs_without_dominant_donor.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


EXCLUDED_DONOR = "GSE183852::TWCM-13-80"
DEFAULT_OUTPUT_ROOT = (
    "new_result2_aging/08_2_combined_LEC_recluster_no_dominant_donor/dataset13"
)
DEFAULT_OUTPUT_H5AD = (
    "08_2_combined_final_LEC_reclustered_no_GSE183852_TWCM-13-80.h5ad"
)


def has_option(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


def main() -> int:
    forwarded = sys.argv[1:]
    engine = Path(__file__).resolve().with_name(
        "08_combine_recluster_final_LEC_age_quartiles.py"
    )
    if not engine.exists():
        raise FileNotFoundError(f"Step 08 engine not found: {engine}")

    if has_option(forwarded, "--exclude_donors"):
        raise ValueError(
            "Step 08.2 has a fixed donor exclusion; do not pass --exclude_donors."
        )

    command = [sys.executable, str(engine), "--exclude_donors", EXCLUDED_DONOR]
    if not has_option(forwarded, "--output_root"):
        command.extend(["--output_root", DEFAULT_OUTPUT_ROOT])
    if not has_option(forwarded, "--out_h5ad"):
        command.extend(["--out_h5ad", DEFAULT_OUTPUT_H5AD])
    if not has_option(forwarded, "--output_prefix"):
        command.extend(["--output_prefix", "08_2"])
    command.extend(forwarded)

    print("Step 08.2 donor-exclusion sensitivity analysis")
    print(f"Excluded donor: {EXCLUDED_DONOR}")
    print("Delegating the complete workflow to the Step 08 engine.")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
