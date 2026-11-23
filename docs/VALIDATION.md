# Validation performed on 21 November 2026

Validation ran in an isolated temporary checkout. The upload folder contains
source code, small provenance inputs and reports; generated test figures/data
and large external inputs are outside it.

| Check | Result and evidence |
|---|---|
| Original preservation | 1,377 preexisting code/documentation/configuration files match their pre-update hashes; `provenance/original_preservation.json` |
| Prior pipeline selection | All 135 previously selected original source hashes match; no changed core analysis was overlooked |
| Code package | Python syntax, code/helper/asset hashes, root functions, declared inputs and workflow targets; `provenance/bundle_validation.json` |
| Current final figures | All 16 match the existing November 9 rendered artwork; 15 whole figures and individual 4B–G retain input bytes; `current_figures_regression.json` |
| Figure 4A | Complete JPEG stream preserved exactly; schematic/artwork outside the declared edit area retained; translated C–G checked against identical placement reference |
| Figure formatting | 107 artwork pages passed pixel, image-payload and retained-text checks; `current_formatting_regression.json` |
| Graphical abstract | Same content inventory, 180 × 110 mm, minimum 8 pt; included in formatting report |
| Figure 4B calibration | Current extracted matching/calibration functions rerun against all 180 acquired fields; calibrated panel render matches existing B; `current_calibration_regression.json` |
| Figure 4E selection | Full primary candidate rankings and three selected cell IDs match existing results; all 12 candidate crops traced to original pixels; heatmap retained; `current_representative_regression.json` |
| Table S8 | 35 data/source worksheets verified; all regenerated workbook archive parts match the existing verified delivery; `current_table_regression.json` |
| Data S10 | 577 documented duplicate cells removed; 4,866 cells/60 wells retained; 186 measurements match existing analysis; all 84 corrected export files match the existing delivery |
| Data S8 | Full 596 MB source rerun: 19 workbooks, 186 data worksheets, 180 source tables, 13,824,109 rows; exact XML/hash checks and worksheet map match existing delivery; `current_data_s8_regression.json` |
| Environments | Both Python environments passed `pip check`; current packaging package versions recorded in `environments/publication_validation_versions_20261121.json` |

Regression reports named above are in `provenance/`. PDF/ZIP container bytes may
differ because of generated identifiers or timestamps; comparisons use original
JPEG streams, unchanged input PDFs, decoded rendered artwork, worksheet XML,
and measurement export bytes as appropriate. The Figure 4 layout check records
small rasterizer rounding differences against direct original rendering and
zero differences against an identically placed reference; it does not conceal
that distinction.

S7 formatting used the checksummed recorded crop calibration. Its optional
`--verify-native-images` path was not executed in the full formatting run;
the report records that explicitly. Figure 4B's acquisition matching and
Figure 4E's native-pixel tracing were independently rerun from read-only
original files, with all outputs isolated.

Full single-cell processing, GSEA/SCENIC inference, R/MultiNicheNet and a new
Cell Painting state/response fit were not rerun in this update. Those unchanged
sources and their executed settings are retained. The Data S8 check repackages
existing MultiNicheNet results; it is not new statistical inference. These
checks do not resolve unavailable data, manuscript metadata or figure-quality
limitations described in `PUBLICATION_ITEMS.md`.

`python tools/validate_release.py` checks package integrity without scientific
imports. `--release` also checks the author-supplied publication metadata and
is expected to report incomplete fields until those are filled. Missing
external inputs on a clean code-only checkout are reported separately.
