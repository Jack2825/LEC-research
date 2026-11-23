# Cardiac lymphatic endothelial cells: manuscript analysis code

Code package updated **21 November 2026** for
[Jack2825/LEC-research](https://github.com/Jack2825/LEC-research).
The current delivery follows the **9 November figure revision**, the
**8 October verified Data S10/Table S8**, and the **28 October Data S8 split**.
Original project code and the previous GitHub package are preserved separately.

The core analysis scripts are unchanged from the previously curated release,
as verified against their original source checksums. New publication scripts
have repository-relative inputs, explicit dependencies, and isolated outputs.
Superseded Figure 4 photo-cropping code, old final assemblers, one-time migration
scripts, and copied historical code archives are excluded. A few still-used
functions are extracted from dated files and identified in the code manifest.

## Start here

```sh
python tools/validate_release.py
python tools/run_pipeline.py --list
python tools/check_inputs.py --group analysis
```

The input check reports missing external data on a code-only checkout.
See [reproduction](docs/REPRODUCING.md), [data access](docs/DATA_ACCESS.md),
[environments](environments/README.md), and [upload instructions](UPLOAD.md).

## Code map

| Folder | Current role |
|---|---|
| `scripts/` | Six required converters/standardizers for datasets 08, 11 and 13 |
| `new_script2_aging/` | Aging QC, LEC states, donor analyses, GSEA, SCENIC and MultiNicheNet |
| `new_script2_disease/` | Disease analyses and active GSE271676 GeoMx stages |
| `cell_painting_LEC_P3_P9_expanded_20260916/` | 97-feature state analysis: 4,866 cells, 60 wells |
| `Cell_Painting_LEC_P3_P9_figures/script/` | 98-feature primary response: 1,626 cells, 20 wells |
| `new_script2/` | Required upstream panel generators and supplementary result tables |
| `representative_images/script/` | Current Fig. 4E selection, original-pixel tracing and documented candidate review |
| `CVR_figure_revision_20261103/script/` | Current numbering, panel letters, S7 scale bars and graphical-abstract typography |
| `figure_calibration/script/` | Current Fig. 4B ND2 matching and 100 µm scale bars; no superseded tube cropping |
| `CVR_figure_revision_20261109/script/` | Final assembly with the complete supplied Figure 4A JPEG |
| `CVR_Data_S10_Table_S8_verified_20261008/script/` | Verified removal of previously identified duplicate records and Table S8 checks |
| `Data_S8_split_Excel_20261028/script/` | Lossless Data S8 worksheet packaging into 19 Excel workbooks |
| `provenance/` | Source selection, exclusions, changes, hashes, inputs and verification records |

Dates in directory names identify the actual source revisions. The input PDFs
are retained artwork dependencies, not historical scripts. Final assembly uses
checksummed frozen inputs; it does not pretend to reconstruct experimental
measurements from artwork. See the [figure map](docs/FIGURE_MAP.md).

## Scientific scope

Aging uses 1,119 LECs, 48 LEC-positive donors and 11 states. Disease uses 6,291
LECs, 95 LEC-positive donors and nine states (Leiden resolution 0.6).
Comparisons remain within disease datasets. The main disease threshold is
25 LECs per donor; HFpEF has separate exploratory 10-cell and sensitivity
15-cell tiers. Cells are not independent donor replicates. State-marker
GSEA/SCENIC describe states rather than direct disease/age contrasts.

The selected GeoMx AOIs contain only two Control donors, so those program
scores remain descriptive. The Cell Painting state and primary-response
feature spaces remain distinct. Data S10 retains 4,866 cells after removal
of 577 provenance-confirmed wrong-folder duplicates; retained values are
unchanged. Channel annotation, segmentation, plasma independence and missing
experimental numerical-source limitations remain documented.

## Release status

The package has been checked locally; nothing was uploaded or changed on GitHub.
The current figure and data packaging was rerun in an isolated checkout;
full single-cell, R/MultiNicheNet and pySCENIC inference was not rerun.
See [validation scope](docs/VALIDATION.md) for the exact checks.

Author/title metadata, the code licence, a release identifier and study-data
access details remain pending in `config/publication_metadata.json`.
`python tools/validate_release.py --release` reports those pending fields;
the ordinary check assesses code-package integrity. See
[publication items](docs/PUBLICATION_ITEMS.md), [citation](CITATION.md) and
[licensing status](LICENSE_STATUS.md).
