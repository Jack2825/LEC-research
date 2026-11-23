# Inputs and data access

The following accessions are evidenced by the current local dataset registry
and executed donor mappings. Access terms must be respected independently of
this repository's eventual code licence.

| Input | Source / access identifier | Local target |
|---|---|---|
| DCM/HCM/NF dataset01 | [Broad SCP1303](https://singlecell.broadinstitute.org/single_cell/study/SCP1303) | `Data/converted/human_dcm_hcm_scportal_03.17.2022.h5ad` |
| DoxCM/NF dataset08 | [GEO GSE292067](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE292067) | Raw `GSE292067_DoxCM_Final_cleanup_used_for_paper.rds`, then included converters |
| HFpEF/NF dataset11 | [Broad SCP3342](https://singlecell.broadinstitute.org/single_cell/study/SCP3342) | Raw H5AD beneath `Data/raw/dataset_11_HFpEF_SCP3342/`, then included standardizer |
| Compiled aging dataset13 | Exact author-supplied `heart_sn_rna_5000_feature.rds`; deposit/access information pending | `Data/raw/dataset_13_Age/heart_sn_rna_5000_feature.rds` |
| Disease GeoMx | [GEO GSE271676](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE271676) | `resources/spatial_transcriptomic/GSE271676_RAW/` |

The Broad study titles/records were checked during the September preparation. GEO accession
values above come from project records; automated access to those specific
GEO records was blocked, so complete citation metadata was not fabricated.

The current aging donor-assignment table contains five component projects:
[EGAS00001006374](https://ega-archive.org/studies/EGAS00001006374),
[ERP123138](https://www.ebi.ac.uk/ena/browser/view/ERP123138),
[GSE156703](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156703),
[GSE183852](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183852) and
[GSE224995](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE224995).
These do not uniquely identify the compiled 5,000-feature RDS or document its
assembly. The available code starts from that RDS. A public or controlled-access
route to this exact input, or its upstream construction code, is still needed
for independent reproduction. EGA data may require managed access.

Dataset01 uses the downloaded processed H5AD directly. Copy the exact named
file to its `Data/converted/` target. Dataset08 requires the R export, Python
H5AD build and standardization scripts, in that order. Dataset11 requires its
standardizer. Dataset13 requires the full-RDS R exporter then the `00b` Python
standardizer. The term "full" means all available cells/genes in that supplied
RDS; the recorded aging state object has 5,000 genes, not an unrestricted genome.

Cell Painting requires the 84 original CellProfiler CSVs under `Data/vivo/Flu P3`
and `Data/vivo/Flu P9`, preserving their folder names. Their exact expected
checksums are bundled in `inputs/verified_raw_export_hashes.csv` inside the
expanded Cell Painting project. The small verified registry is bundled; it
freezes 4,866 retained cells and the prior exclusion of 577 wrong-folder copies.
Original ND2 files are needed for native image QC and crop verification. The
repository does not contain a CellProfiler segmentation pipeline because none
was supplied in the inspected project. Missing raw segmentation masks are not
reconstructed or inferred.

`provenance/required_assets.csv` lists model/database filenames, sizes and hashes:
Healthy_Adult_Heart CellTypist model, MSigDB human 2026.1 GMT collections,
pySCENIC hg38 rankings/motif annotations/TF list, and GeoMx input files. Obtain
these versions from their providers (CellTypist, MSigDB and cisTarget); third-party
databases are not relicensed or bundled. MultiNicheNet reports retain their
documented NicheNet network download URLs and caching logic.

Current publication inputs are listed in
`provenance/publication_input_manifest.csv`, including their source locations,
repository-relative targets, sizes, SHA-256 and whether bundled. Groups are:

| Group | Purpose | Approximate input size |
|---|---|---:|
| `figures` | Final November 9 exact-artwork route | 115 MB |
| `formatting` | Current numbering, S7 and graphical-abstract formatting | 147 MB |
| `tables` | Corrected Data S10 and verified Table S8 | 93 MB |
| `data_s8` | Full unsplit MultiNicheNet workbook | 596 MB |
| `calibration` | Fig. 4B original acquisitions, corrected images and CP mapping | 6.14 GB |
| `representative` | Fig. 4E saved state/image inputs and original ND2 | See manifest; ND2 overlap calibration |

The small retained-cell/exclusion records and S7 calibration are bundled.
Publication artwork, Excel/ZIP collections, raw images and expression objects
are external. The local staging helper can copy the exact declared inputs from
the original project; it is not a public data-download service. A data deposit
or accurate access statement is still required for outside researchers.

The updated Data S10 is provenance-filtered, with unchanged retained values;
the unfiltered original archive must not be described as the final corrected
release. The raw-export hashes in the expanded analysis remain intentional:
that pipeline reads original exports and applies the saved retained registry.
Do not substitute filtered exports under those original filenames and hashes.

Final Fig. 4B has a verified acquisition scale (0.324081161957359 µm/pixel);
S7A uses the same recorded physical pixel scale and its exact crop extents.
The tube JPEG carries the supplied 1 mm bar; no calibration for its other
views is inferred. Separate tube/vascular/cardiac numerical tables were not
supplied, so their statistics cannot be recomputed from the inherited artwork.
