# Expanded Cell Painting analysis: plan fixed before treatment results

Created 2026-09-16. Existing projects are preserved; their non-ND2 files have a SHA-256 manifest. All writes from this project stay in this new folder. No raw exports or original image copies are placed here.

## Scope and units

Rebuild all exported cell measurements from the five object tables using ImageNumber and Parent_nuclei. The verified corrected registry supplies retained cell identities and experimental labels, not the analysis feature panel. Confirm 4,866 cells, 60 physical wells, and removal of precisely the previously verified 577 wrong-folder copies. Five wells per condition are treated as separate replicates at the user's request; donor identities are not independently documented. H1N1 denotes plasma from H1N1-infected mice.

The supplied strategy is background guidance, not a predetermined biological conclusion. There are 30 available P9 wells out of 36 planned; G03-G08 are absent. Passage is confounded with acquisition date and segmentation settings. Prior outputs are never regenerated or edited here.

## Features and QC

Inventory the 186 measurements and the 155 bearing AreaShape/Intensity names. Some AreaShape fields are absolute coordinates: exclude these as well as absolute orientation from biological distances. Keep coordinates for technical diagnostics. No actin/Golgi features or mitochondrial network morphology are invented. ER-derived cell shape is a secondary-object cell-region proxy.

Reject features with >1% nonfinite values, >=99% identical values, or zero robust spread. Missing values, if present below the limit, use an equal-well median. Nonnegative unbounded variables use log1p(x / equal-well median positive x); bounded shape ratios remain linear. Standardize by equal-well median and 1.4826 MAD, with IQR/1.349 fallback. Greedily prune absolute equal-well weighted Spearman correlations >0.90, prioritizing interpretable measurements over Zernike descriptors. Every decision is exported. Use equal total weight per well for PCA and k-means.

Flag edge-touching objects, saturated maxima, and any |robust z|>8. Do not remove large phenotypic deviations solely because they are rare. Primary analysis retains valid cells. Edge/saturation exclusion is a separate sensitivity; it does not certify segmentation quality. Read representative original images for review without copying ND2 files.

## Continuous and minority-cell responses

Reference each passage to its own Control/None wells. The primary distance is regularized Mahalanobis distance in the full retained standardized feature space, using equal-well control means and covariances and 20% shrinkage toward an isotropic covariance. No chi-square assumption is used for responder thresholds.

Fit five models, each withholding an entire control well. Each control cell is scored only by the model excluding its own well. Pool those held-out scores with equal well weight to define empirical 90th, 95th (primary), and 97.5th percentiles. Score each nonreference cell with all five models. Report its mean normalized distance, the fraction of models calling it a responder, and the per-well average responder fraction over those models. A consensus responder exceeds the threshold in >=4/5 models; it is a screening flag, not a validated state. Control cells have only one out-of-fold evaluation and cannot be assigned the same consensus status.

Primary tests: H1N1 versus Control without intervention in each passage, and VEGF-C/UC2288 versus no intervention under H1N1 in each passage (six contrasts). Endpoints are 95%-tail fraction, median distance, and 90th-percentile distance (18 tests; one BH family). Exact 5-versus-5 well-label permutations use all 252 assignments. For H1N1-versus-reference tests, refit the control models AND threshold under every assignment; fixed-score permutations would be inappropriate. Intervention tests use an external common Control/None reference independent of the compared groups.

Compartment 95%-tail tests form a separate 30-test BH family (five measurement families x six contrasts). Feature drivers use per-well medians, 10th and 90th quantiles with a single BH family over all tested features, summaries, and contrasts. No cell-level treatment P values. Percentile intervals from 5,000 well resamples are conditional on the fitted feature representation/reference; they do not include reference-training uncertainty. Full leave-one-reference-well-out re-estimation and alternate metrics quantify that uncertainty separately. All thresholds are exploratory design choices, not universal biological cutoffs.

Sensitivity: 10% and 50% covariance shrinkage, diagonal standardized distance, 90% and 97.5% tail thresholds, removal of edge/saturated cells, and leave-one-control-well-out reference reconstruction. Compare direction/effect sizes without selecting the lowest P value. A 95th-percentile flag is unusual morphology relative to these controls, not evidence of damage or dysfunction.

## State discovery and embedding

Weighted PCA and k-means in the common space and separately within each passage. Evaluate k=2..6, PCA variance 85/90/95%, and three random seeds. For each k at 90% PCA, 40 hierarchical well/cell bootstraps refit robust scaling and PCA, and all leave-one-well-out fits refit those steps. Feature identities/transforms/correlation selection are frozen, so stability is conditional on that feature panel. Report global ARI AND per-state best-match Jaccard; global agreement alone can conceal an unstable minority state. Compare morphology-only, no-nuclear-intensity and QC-filtered representations. HDBSCAN in PCA space supplies a density-based sensitivity that may leave cells unassigned.

State screens: median bootstrap and leave-one-well-out ARI >=0.80, per-state median Jaccard >=0.75, representation by >=2 cells in >=3 wells, and no well contributing >40% of equal-well state mass. Examine cross-parameter and feature-panel stability. These are reproducibility screens, not biological validation. Do not select k using treatment separation or significant treatment tests. Retain all candidate partitions; state proportions across candidates are exploratory and use one combined BH family. Allow the conclusion that no discrete states are supported.

UMAP uses features alone: neighbors 15/30/60, min_dist 0.1/0.5, three seeds (18 settings). Display all settings, neighbor preservation and seed agreement. Use one fixed default (neighbors=30,min_dist=0.5,first seed) for group/well/state overlays. Clusters are learned in PCA space, not from gaps in 2-D UMAP.

## Interpretation

Prioritize reproducible well-level tail changes and interpretable feature changes. Descriptive features of responders are selected using the same phenotype matrix and are not independent mechanistic validation. Image crops lack exported segmentation masks; manual review can reveal concerns but cannot certify every cell. No claims of senescence, mitochondrial dysfunction, adaptive/maladaptive states, or functional rescue without independent evidence.

Method context: [UMAP cautions about density and artificial splits](https://umap-learn.readthedocs.io/en/latest/clustering.html); [covariance regularization](https://scikit-learn.org/stable/modules/covariance.html); [single-cell Cell Painting heterogeneity](https://www.nature.com/articles/s41467-019-10154-8); [SPACe single-cell phenotypic analysis](https://www.nature.com/articles/s41467-024-54264-4). These motivate the approach; numerical thresholds above are prespecified project choices.
