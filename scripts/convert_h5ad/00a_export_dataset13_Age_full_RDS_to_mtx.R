# 00a_export_dataset13_Age_full_RDS_to_mtx.R
# Export dataset13 full RDS to Matrix Market + CSV for Python h5ad build.
# Run from project root:
#   Rscript "new_scripts/00a_export_dataset13_Age_full_RDS_to_mtx.R"

suppressPackageStartupMessages({ library(Matrix) })

PROJECT_DIR <- normalizePath(Sys.getenv("LEC_PROJECT_ROOT", unset = getwd()), winslash = "/", mustWork = TRUE)
RDS_PATH <- file.path(PROJECT_DIR, "Data/raw/dataset_13_Age/heart_sn_rna_5000_feature.rds")
EXPORT_DIR <- file.path(PROJECT_DIR, "Data/converted/dataset_13_Age_5000feature_rds_export")
MERGE_ALL_OBJECTS <- TRUE

dir.create(EXPORT_DIR, recursive = TRUE, showWarnings = FALSE)

message("Input RDS: ", RDS_PATH)
message("Export dir: ", EXPORT_DIR)
if (!file.exists(RDS_PATH)) stop("Cannot find RDS: ", RDS_PATH)

if (!requireNamespace("Seurat", quietly = TRUE)) {
  stop("Seurat is required. In R, install it first: install.packages('Seurat')")
}
suppressPackageStartupMessages(library(Seurat))

safe_name <- function(x) {
  x <- gsub("[^A-Za-z0-9_.+-]+", "_", as.character(x))
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  ifelse(nchar(x) == 0, "object", x)
}

is_seurat <- function(x) inherits(x, "Seurat")

flatten_seurat <- function(x, prefix = "root") {
  out <- list()
  if (is_seurat(x)) {
    out[[prefix]] <- x
    return(out)
  }
  if (is.list(x)) {
    nms <- names(x)
    if (is.null(nms)) nms <- paste0("item", seq_along(x))
    for (i in seq_along(x)) {
      nm <- nms[[i]]
      if (is.na(nm) || nm == "") nm <- paste0("item", i)
      out <- c(out, flatten_seurat(x[[i]], paste0(prefix, "__", safe_name(nm))))
    }
  }
  out
}

get_counts_meta <- function(obj, obj_name) {
  assay_use <- if ("RNA" %in% Seurat::Assays(obj)) "RNA" else Seurat::DefaultAssay(obj)
  message("Extracting: ", obj_name, " | assay: ", assay_use)
  counts <- tryCatch(Seurat::GetAssayData(obj, assay = assay_use, layer = "counts"), error = function(e) NULL)
  if (is.null(counts)) {
    counts <- tryCatch(Seurat::GetAssayData(obj, assay = assay_use, slot = "counts"), error = function(e) NULL)
  }
  if (is.null(counts)) stop("Could not extract counts from: ", obj_name)
  if (!inherits(counts, "dgCMatrix")) counts <- as(counts, "dgCMatrix")
  meta <- obj@meta.data
  meta$original_cell <- rownames(meta)
  meta <- meta[colnames(counts), , drop = FALSE]
  list(counts = counts, meta = meta, assay = assay_use)
}

message("Reading RDS...")
rds_obj <- readRDS(RDS_PATH)
message("Top-level class: ", paste(class(rds_obj), collapse = ";"))
objs <- flatten_seurat(rds_obj)
if (length(objs) == 0) stop("No Seurat objects found inside RDS.")
message("Seurat objects found: ", length(objs))

data <- list()
candidates <- list()
for (nm in names(objs)) {
  z <- tryCatch(get_counts_meta(objs[[nm]], nm), error = function(e) { warning(conditionMessage(e)); NULL })
  if (!is.null(z)) {
    data[[nm]] <- z
    candidates[[length(candidates) + 1]] <- data.frame(
      object_name = nm,
      assay_used = z$assay,
      n_genes = nrow(z$counts),
      n_cells = ncol(z$counts),
      n_nonzero = length(z$counts@x),
      stringsAsFactors = FALSE
    )
  }
}
if (length(data) == 0) stop("No usable Seurat count matrices extracted.")

candidate_table <- do.call(rbind, candidates)
candidate_table <- candidate_table[order(candidate_table$n_cells, decreasing = TRUE), ]
write.csv(candidate_table, file.path(EXPORT_DIR, "rds_object_candidates.csv"), row.names = FALSE)
message("Candidate objects:")
print(candidate_table)

if (MERGE_ALL_OBJECTS && length(data) > 1) {
  message("Merging all objects by shared genes...")
  shared_genes <- Reduce(intersect, lapply(data, function(z) rownames(z$counts)))
  if (length(shared_genes) < 100) stop("Too few shared genes across objects: ", length(shared_genes))
  message("Shared genes: ", length(shared_genes))
  counts_list <- list()
  meta_list <- list()
  used_cells <- character(0)
  for (nm in names(data)) {
    z <- data[[nm]]
    counts <- z$counts[shared_genes, , drop = FALSE]
    old_cells <- colnames(counts)
    new_cells <- old_cells
    if (any(new_cells %in% used_cells) || any(duplicated(new_cells))) {
      new_cells <- paste0(safe_name(nm), "__", old_cells)
    }
    colnames(counts) <- new_cells
    meta <- z$meta[old_cells, , drop = FALSE]
    meta$cell <- new_cells
    meta$original_cell <- old_cells
    meta$rds_object_name <- nm
    meta$rds_assay_used <- z$assay
    rownames(meta) <- new_cells
    counts_list[[nm]] <- counts
    meta_list[[nm]] <- meta
    used_cells <- c(used_cells, new_cells)
  }
  counts_all <- do.call(Matrix::cBind, counts_list)
  meta_all <- do.call(rbind, meta_list)
} else {
  nm <- candidate_table$object_name[[1]]
  message("Using largest object only: ", nm)
  z <- data[[nm]]
  counts_all <- z$counts
  meta_all <- z$meta
  meta_all$cell <- colnames(counts_all)
  meta_all$rds_object_name <- nm
  meta_all$rds_assay_used <- z$assay
  rownames(meta_all) <- meta_all$cell
}

meta_all <- meta_all[colnames(counts_all), , drop = FALSE]

message("Final matrix genes x cells: ", nrow(counts_all), " x ", ncol(counts_all))
message("Nonzero entries: ", length(counts_all@x))

Matrix::writeMM(counts_all, file.path(EXPORT_DIR, "counts_genes_by_cells.mtx"))
write.csv(data.frame(gene_symbol = rownames(counts_all), gene_original = rownames(counts_all)), file.path(EXPORT_DIR, "genes.csv"), row.names = FALSE)
write.csv(data.frame(cell = colnames(counts_all)), file.path(EXPORT_DIR, "cells.csv"), row.names = FALSE)
meta_out <- meta_all
meta_out$cell <- rownames(meta_out)
meta_out <- meta_out[, c("cell", setdiff(colnames(meta_out), "cell")), drop = FALSE]
write.csv(meta_out, file.path(EXPORT_DIR, "metadata.csv"), row.names = FALSE)

markers <- c("PECAM1", "VWF", "CDH5", "PROX1", "LYVE1", "PDPN", "FLT4", "CCL21", "MMRN1", "TBX1")
marker_check <- data.frame(marker = markers, present = toupper(markers) %in% toupper(rownames(counts_all)))
write.csv(marker_check, file.path(EXPORT_DIR, "marker_check.csv"), row.names = FALSE)

summary <- data.frame(
  dataset_id = "dataset_13_Age",
  input_rds = RDS_PATH,
  export_dir = EXPORT_DIR,
  merge_all_objects = MERGE_ALL_OBJECTS,
  n_genes = nrow(counts_all),
  n_cells = ncol(counts_all),
  n_nonzero = length(counts_all@x),
  counts_mtx = file.path(EXPORT_DIR, "counts_genes_by_cells.mtx"),
  genes_csv = file.path(EXPORT_DIR, "genes.csv"),
  cells_csv = file.path(EXPORT_DIR, "cells.csv"),
  metadata_csv = file.path(EXPORT_DIR, "metadata.csv")
)
write.csv(summary, file.path(EXPORT_DIR, "export_summary.csv"), row.names = FALSE)
message("Done exporting full RDS.")
