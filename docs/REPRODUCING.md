# Reproduction

Use this repository as a new analysis workspace. Root-resolution functions point
to this checkout even if a historical project is nearby. Do not copy the entire
old `scripts/` or `new_scripts/` directories into it. Outputs are ignored by Git.
Run from the repository root, without `python -O`.

## 1. Verify and prepare inputs

```sh
python tools/validate_release.py
python tools/check_inputs.py --group analysis --sha256
```

Install the separate [environments](../environments/README.md). Restore the data
and external resources from [DATA_ACCESS.md](DATA_ACCESS.md). The input checker
is a declared-asset check, not an exhaustive validation of every downstream
intermediate file. Stage scripts perform their own schema/count/cohort checks.

Build active source objects, if not restoring the exact processed objects:

```sh
Rscript scripts/convert_h5ad/00_convert_dataset08_DoxCMP_RDS_to_export.R
python scripts/convert_h5ad/00_build_dataset08_DoxCMP_h5ad.py
python scripts/standardization/01_standardize_dataset08_DoxCMP.py
python scripts/standardization/01_standardize_dataset11_HFpEF_SCP3342.py
Rscript scripts/convert_h5ad/00a_export_dataset13_Age_full_RDS_to_mtx.R
python scripts/standardization/00b_build_dataset13_Age_full_h5ad_standardize.py
```

Place the dataset01 processed H5AD directly at its documented target. The R
converters use `LEC_PROJECT_ROOT`, defaulting to the current directory.

## 2. Scientific analyses

```sh
python tools/run_pipeline.py --list
python tools/run_pipeline.py --branch aging
python tools/run_pipeline.py --branch disease
python tools/run_pipeline.py --branch geomx
```

Without `--execute` these commands show the ordered commands only. After checking
inputs, add `--execute` to execute a branch, or select one exact `--step` from the
list. For example:

```sh
python tools/run_pipeline.py --step aging_03 --execute
python tools/run_pipeline.py --step disease_08 --execute
```

The runner does not silently skip completed files or remove results. Existing
script-specific overwrite protections still apply. Use a fresh checkout/work
directory for an independent rerun. Raw single-cell processing can involve tens
of gigabytes and substantial RAM; no new runtime estimates are claimed.

Aging order: 03–08, identity/dominant-donor checks 08.1–08.2, equal-donor quartiles
08.3, donor analyses 08.4–08.7, GSEA 09, SCENIC preparation/PowerShell execution
10, SCENIC summary 11, MultiNicheNet R Markdown 13. Disease order: 03–08,
08.1–08.4, 09–11, MultiNicheNet 12. GeoMx is 13.1–13.4. Executed run records
are in `provenance/executed_runs/`; do not substitute a smoke-test cohort.

## 3. Cell Painting

```sh
python tools/run_pipeline.py --branch cell_painting --execute
python tools/run_pipeline.py --branch cell_painting_primary --execute
```

The expanded stages reconstruct measurements from verified raw exports, compute
responses, fit the existing state-analysis specification, review ND2 images,
render and validate. Stage 04 requires ND2 images. If deliberately omitted,
run individual stages and record that image validation was not performed.
Native-image figure selection later still requires its verified image inputs.

The primary branch runs C preprocessing/PCA, D response tests, E source import
and saved-state visualization, F proportions, and G exploratory summaries.
It does not recompute/rename the saved states within the 98-feature primary
representation. The primary image panel uses supplied image inputs. The current tube
artwork is handled only by the November 9 final-figure script. Do not use `--channel-labels` to imply an unverified mapping has
been resolved; retain the inherited display and its disclosed limitation.

## 4. Current publication stages

The unchanged `new_script2/` panel generators consume completed analysis
outputs. Their historical figure IDs map to final manuscript IDs in
[FIGURE_MAP.md](FIGURE_MAP.md). Generate the required disease
`FigS04_disease_all_state_effects.py` source table before S5. The upstream
supplement builder creates intermediate workbooks; its disease GeoMx Data S10
corresponds to manuscript Data S9, not the Cell Painting Data S10 below.

All publication inputs are explicit in
`provenance/publication_input_manifest.csv`. Restore deposited files to its
repository-relative `path` column. If you have the original local study
project, this optional helper verifies and copies only the declared files:

```sh
python tools/stage_publication_inputs.py --group figures --from-project /path/to/original/project
python tools/stage_publication_inputs.py --group figures --from-project /path/to/original/project --copy
```

The first command shows a plan. Neither command writes to the source project.
Repeat with `formatting`, `tables`, `data_s8`, `calibration` or `representative`
only for the stages needed. Large files belong in the data deposit and remain
ignored by Git. Do not upload the resulting analysis workspace wholesale.

### Final figures: November 9

```sh
python tools/check_inputs.py --group figures --sha256
python tools/reproduce_final_figures.py
```

Output: `CVR_figure_revision_20261109/result/figures/` contains all 16 final
PDFs. The complete original 2,695 × 1,828 JPEG is embedded unchanged in Fig. 4A.
The schematic is retained; C–G move down 100 PDF points without rescaling.
The other 15 whole figures and individual 4B–G retain their recorded bytes.
This is the exact artwork route used by the current final script, and requires
the declared frozen base artwork. PNG files are previews.

### Current upstream formatting and calibration

```sh
python tools/check_inputs.py --group formatting --sha256
python tools/reproduce_final_figures.py --formatting
```

This rebuilds numbering/panel letters, calibrated S7A overlays, full-size panels,
and the graphical abstract in `CVR_figure_revision_20261103/result/`.
Its Figure 4 is an intermediate base; use the November 9 output for final Fig. 4.
The current graphical abstract is 180 × 110 mm with text at least 8 pt.
Original Arial fonts are required; see the environment instructions.

S7 crop geometry/calibration is recorded with hashes. To independently verify
the matching acquisition files, restore the original ND2 paths and add
`--verify-native-images` to the formatting command. Without that option the
report explicitly records that native acquisition calibration was not rechecked.
`CVR_figure_revision_20261103/script/prepare.py --refresh-calibration`
can regenerate the crop-calibration audit into `result/calibration/` from
the expanded analysis outputs and original ND2; it does not overwrite the
bundled reference calibration.

For the Fig. 4B acquisition-matching and bar-generation procedure:

```sh
python figure_calibration/script/figure4b.py
python figure_calibration/script/figure4b.py --panel /path/to/unbarred/Fig04B.pdf
```

Restore the `calibration` group first. This compares the image against all
180 acquired fields, verifies the CellProfiler image/frame mapping and physical
scale, and writes its audit locally. The second form adds bars to an unbarred
panel. The final figure route retains the already verified bar artwork.
The CY5/TRITC compartment interpretation remains unresolved.

For current Fig. 4E image selection, restore `representative` inputs and run:

```sh
python representative_images/script/panel_E_representative_images.py --prepare --render
```

This uses saved state assignments/features and the original ND2 pixels. It
preserves the recorded visual-review choices and does not refit states.

### Corrected Data S10 and verified Table S8: October 8

```sh
python tools/check_inputs.py --group tables --sha256
python CVR_Data_S10_Table_S8_verified_20261008/script/prepare_verified_package.py
```

Output is the same module's `result/` directory. This removes only the 577
previously identified duplicate cell records, checks all 186 retained raw
measurements against existing analysis input, verifies 35 Table S8 worksheets,
and preserves scientific worksheet XML. Keep the exact-precision companion
ZIP with Table S8. The original unfiltered archive is an input to this step,
not the current Data S10 deliverable. Table S8's frozen artwork provenance
predates the November tube JPEG; the final-figure input manifest separately
identifies that replacement. No missing tube measurements are inferred.

### Data S8: October 28 lossless split

```sh
python tools/check_inputs.py --group data_s8 --sha256
python Data_S8_split_Excel_20261028/script/split_data_s8.py
```

The complete 596,264,080-byte source workbook becomes 19 data workbooks below
50 decimal MB each, plus an index. All 186 data worksheets, 180 source tables
and 13,824,109 data rows are retained. README and Source_Index repeat unchanged
in every part. The script checks exact worksheet XML hashes, ZIP integrity and
readability. It refuses an existing output directory. For a separate rerun use
`--output /path/to/new/result`; `--source` selects another explicit workbook.
The 50 MB split cap is this project's packaging setting, not a claim that the
journal mandates this exact size. A total upload limit is not reduced by splitting.

`python tools/run_pipeline.py --branch publication` lists the three final
publication commands; add `--execute` only after restoring the relevant inputs.
Independent scientific reproduction still requires the exact source data and
recorded environments. The validation report distinguishes that from packaging.
