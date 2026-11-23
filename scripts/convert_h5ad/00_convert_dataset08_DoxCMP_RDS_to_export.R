suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

project_dir <- normalizePath(Sys.getenv("LEC_PROJECT_ROOT", unset = getwd()), winslash = "/", mustWork = TRUE)
dataset_id <- "dataset_08_DoxCMP"

raw_dir <- file.path(project_dir, "Data/raw", dataset_id)

input_file <- file.path(
  raw_dir,
  "GSE292067_DoxCM_Final_cleanup_used_for_paper.rds"
)

export_dir <- file.path(
  raw_dir,
  paste0(dataset_id, "_export")
)

dir.create(export_dir, recursive = TRUE, showWarnings = FALSE)

cat("Input file:\n")
cat(input_file, "\n\n")

if (!file.exists(input_file)) {
  stop("Cannot find input file: ", input_file)
}

cat("Reading RDS...\n")
obj <- readRDS(input_file)

if (!"Seurat" %in% class(obj)) {
  stop("Object is not a Seurat object.")
}

cat("\nSeurat object:\n")
print(obj)

cat("\nAssays:\n")
print(Assays(obj))

# Use RNA raw counts, not SCT.
assay_use <- "RNA"

if (!assay_use %in% Assays(obj)) {
  stop("RNA assay not found.")
}

cat("\nUsing assay:\n")
cat(assay_use, "\n")

# Compatible with Seurat v5 and older versions
counts <- tryCatch(
  {
    GetAssayData(obj, assay = assay_use, layer = "counts")
  },
  error = function(e) {
    GetAssayData(obj, assay = assay_use, slot = "counts")
  }
)

cat("\nCounts matrix dimensions genes x cells:\n")
print(dim(counts))

cat("\nWriting counts matrix...\n")
Matrix::writeMM(
  counts,
  file.path(export_dir, "counts_genes_by_cells.mtx")
)

cat("Writing genes.csv...\n")
write.csv(
  data.frame(gene = rownames(counts)),
  file.path(export_dir, "genes.csv"),
  row.names = FALSE
)

cat("Writing cells.csv...\n")
write.csv(
  data.frame(cell = colnames(counts)),
  file.path(export_dir, "cells.csv"),
  row.names = FALSE
)

cat("Writing metadata.csv...\n")
metadata <- obj@meta.data
metadata$cell <- rownames(metadata)

write.csv(
  metadata,
  file.path(export_dir, "metadata.csv"),
  row.names = FALSE
)

# Export dimensional reductions if present
if ("umap" %in% Reductions(obj)) {
  cat("Writing UMAP...\n")
  umap <- Embeddings(obj, reduction = "umap")
  umap_df <- as.data.frame(umap)
  umap_df$cell <- rownames(umap_df)

  write.csv(
    umap_df,
    file.path(export_dir, "umap.csv"),
    row.names = FALSE
  )
}

if ("pca" %in% Reductions(obj)) {
  cat("Writing PCA...\n")
  pca <- Embeddings(obj, reduction = "pca")
  pca_df <- as.data.frame(pca)
  pca_df$cell <- rownames(pca_df)

  write.csv(
    pca_df,
    file.path(export_dir, "pca.csv"),
    row.names = FALSE
  )
}

if ("harmony" %in% Reductions(obj)) {
  cat("Writing harmony...\n")
  harmony <- Embeddings(obj, reduction = "harmony")
  harmony_df <- as.data.frame(harmony)
  harmony_df$cell <- rownames(harmony_df)

  write.csv(
    harmony_df,
    file.path(export_dir, "harmony.csv"),
    row.names = FALSE
  )
}

cat("\nDone exporting dataset08.\n")
cat("Export folder:\n")
cat(export_dir, "\n")