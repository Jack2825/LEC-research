# Environments

The inspected working environments are Python 3.14.6 for analysis/figures and
Python 3.10.11 for pySCENIC. Both passed `pip check` again on 21 November 2026.
`requirements-main.txt` and `requirements-scenic.txt` pin observed direct
dependencies. They are not complete transitive, cross-platform lockfiles.
Install them in separate environments. No dependencies were upgraded during packaging.

```sh
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r environments/requirements-main.txt
py -3.10 -m venv .venv_scenic
.venv_scenic/Scripts/python.exe -m pip install -r environments/requirements-scenic.txt
```

These are Windows commands. On POSIX use `bin/python` and a Python 3.10
interpreter for the SCENIC environment. The supplied SCENIC launchers require
PowerShell and use `.venv_scenic/Scripts/pyscenic.exe`. They record the exact GRN,
cisTarget and AUCell arguments. The rest of the Python code uses portable paths.
Do not run `python -O`: scientific input checks use assertions.

The current R reports specify R 4.5.3. Exact recorded R session information is
in `provenance/executed_runs/`. CRAN dependencies include Matrix, dplyr, tidyr,
tibble, ggplot2, jsonlite, knitr, rmarkdown and pkgbuild. Bioconductor dependencies
include SingleCellExperiment, SummarizedExperiment, S4Vectors, zellkonverter,
rhdf5, HDF5Array, DelayedArray, edgeR, limma, bluster, ComplexHeatmap, IHW,
muscat, scater, scran and sva. NicheNet and MultiNicheNet are supplied by
`saeyslab/nichenetr` and `saeyslab/multinichenetr`.

R Markdown currently retains the original first-run installation/download
options. For an already prepared environment set `install_missing_packages`
and `allow_network_download` to false when rendering. Historical GitHub commit
SHAs were not recorded in the available session logs; a complete R lockfile
cannot be honestly reconstructed. R execution was not available in the
packaging shell; the reports were retained and their code inspected.

The separately recorded Cell Painting versions describe that executed analysis;
the current installed environment snapshot does not prove every earlier analysis
used precisely the same package versions. Random seeds and executed settings are
preserved in the code and recorded run configurations.

Current PDF typography requires the original Arial regular and bold font files.
The formatter locates standard Windows/macOS/Linux installations. Otherwise set
`LEC_ARIAL_REGULAR` and `LEC_ARIAL_BOLD` to licensed local TTF paths. The repository
does not distribute Microsoft font files or silently substitute different font
metrics. Final exact-JPEG assembly uses the fonts already embedded in its input
PDFs and does not require this font lookup.
