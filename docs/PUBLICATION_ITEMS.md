# Journal requirements and remaining author items

Policy reviewed 21 November 2026 against the
[Cardiovascular Research General Instructions](https://academic.oup.com/cardiovascres/pages/General_Instructions).
The journal requires a Data Availability Statement and dataset references with
accession identifiers. Sharing underlying code is strongly encouraged; it does
not prescribe this GitHub folder structure, a particular code licence, or a
software DOI. Code packaging alone cannot certify manuscript acceptance.

The repository provides selected current code, recorded parameters, environments,
input checksums, reproduction commands and validation evidence. The latest
Figure 4B scale is now verified and the graphical abstract has been formatted
to 180 × 110 mm with text at least 8 pt. The earlier unresolved-calibration and
unformatted-abstract statements are therefore superseded.

Items still requiring author completion or manuscript review:

- Complete authors, manuscript title, code rights/licence, release identifier
  and genuine data-access information in `config/publication_metadata.json`.
  A stable archived release is useful; do not invent an archive DOI.
- Provide access to the exact compiled aging RDS or its construction procedure,
  and complete dataset references. Supply the necessary study data/artwork
  deposit or accurate access restrictions; a local staging command is not
  evidence that the data are publicly available.
- Match the manuscript to corrected Data S10: 577 duplicate records removed,
  4,866 cells retained, retained values unchanged. Keep all Data S8 parts/index
  and the Table S8 exact-precision companion ZIP available together.
- CY5/TRITC compartment annotation, missing segmentation masks, State 0 image
  QC flags and plasma/well independence are not resolved by packaging.
- Numerical tube/vascular/cardiac source tables and graph-specific replicate
  counts/tests remain unavailable in the supplied code inputs. The final tube
  JPEG is preserved intact, including its supplied 1 mm bar; this does not
  establish calibration for all its views. Figure 7D scale length remains
  unresolved in the reviewed records. Reconcile the missing Table S7 reference.
- Review all legends, main-figure alt text and final print-scale readability.
  Some inherited raster artwork remains below the recommended resolution;
  a higher export DPI cannot create original image detail. Dense native-size
  panels can become unreadable if reduced excessively. Figure 6 retains eight
  panels, exceeding the journal's recommendation to avoid more than six.

No scientific thresholds, significance, state labels or source measurements were
changed to create an appearance of compliance. No authorship, ethics, funding,
consent, licence, accession or DOI has been fabricated.
