#!/usr/bin/env python3
"""Build journal-ready Supplementary Data S1-S10 and Tables S1-S6.

Each deliverable is a separate Excel workbook.  Existing analysis tables are
copied into worksheets without row filtering.  Files that exceed Excel's row
limit are split across numbered worksheets in the same workbook.  Every
workbook contains a README and a Source_Index worksheet with paths, checksums,
dimensions, and any transformation performed by this packaging script.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT / "new_result2" / "supplementary_data_and_tables"
DATA_DIR = OUTPUT_ROOT / "Supplementary_Data"
TABLE_DIR = OUTPUT_ROOT / "Supplementary_Tables"

# Excel allows 1,048,576 rows per sheet.  One row is reserved for the header.
MAX_DATA_ROWS = 1_048_575
MAX_COLUMNS = 16_384
MAX_CELL_CHARACTERS = 32_767

HEADER_FILL = PatternFill("solid", fgColor="17365D")
HEADER_FONT = Font(color="FFFFFF", bold=True)
README_TITLE_FONT = Font(color="17365D", bold=True, size=14)

TABULAR_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".tsv.gz")


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def tabular_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and any(path.name.lower().endswith(s) for s in TABULAR_SUFFIXES)
    )


def selected_files(root: Path, required_fragments: Sequence[str]) -> list[Path]:
    files = tabular_files(root)
    return [p for p in files if any(fragment.lower() in p.name.lower() for fragment in required_fragments)]


def strip_tabular_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in TABULAR_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def short_sheet_label(path: Path, prefix: str = "") -> str:
    stem = strip_tabular_suffix(path.name)
    stem = re.sub(r"^(?:FigS?\d+[A-Z]?_|\d+(?:_\d+)?_)", "", stem, flags=re.IGNORECASE)
    replacements = {
        "MultiNicheNet": "MNN",
        "normalized_pseudobulk": "normPB",
        "prioritization": "priority",
        "ligand_activities": "ligand_activity",
        "cell_metadata": "cell_meta",
        "counts_percentages": "counts_pct",
        "state_marker_ranking": "marker_rank",
        "state_collection_summary": "collection_summary",
        "state_ranking": "ranking",
        "donor_balanced": "donor_bal",
        "equal_cell_balanced": "equal_cell",
        "leave_one_donor_out": "LODO",
        "leave_one_project_out": "LOPO",
        "upper_quartile": "UQ",
        "prespecified": "prespec",
        "validation": "valid",
        "sensitivity": "sens",
        "summary": "sum",
        "expression": "expr",
        "interactions": "interact",
    }
    for old, new in replacements.items():
        stem = stem.replace(old, new)
    label = f"{prefix}_{stem}" if prefix else stem
    label = re.sub(r"[\\/*?:\[\]]", "_", label)
    label = re.sub(r"_+", "_", label).strip("_ ")
    if len(label) > 31:
        tag = hashlib.sha1(label.encode("utf-8")).hexdigest()[:5]
        label = f"{label[:25]}_{tag}"
    return label or "Data"


def disease_mnn_prefix(path: Path) -> str:
    parts = set(path.parts)
    if "dataset01_DCM_vs_NF" in parts:
        return "DCM"
    if "dataset01_HCM_vs_NF" in parts:
        return "HCM"
    if "dataset08_DoxCM_vs_NF" in parts:
        return "DoxCM"
    if "dataset11_HFpEF_vs_NF_min10" in parts:
        return "HFpEF10"
    if "dataset11_HFpEF_vs_NF_min15" in parts:
        return "HFpEF15"
    return "Overview"


def stage_prefix(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) <= 1:
        return "Overview"
    parent = relative.parts[0]
    match = re.match(r"(\d+_\d+)", parent)
    return match.group(1) if match else parent[:8]


def coerce_excel_value(value):
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (int, bool)):
        return value
    text = str(value)
    if text == "" or text.lower() in {"nan", "na", "n/a", "null", "none"}:
        return None
    if len(text) > MAX_CELL_CHARACTERS:
        raise ValueError(f"Cell text exceeds Excel's {MAX_CELL_CHARACTERS}-character limit")
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if re.fullmatch(r"-?(?:0|[1-9]\d*)", text):
        digits = text.lstrip("-")
        if len(digits) <= 15:
            try:
                return int(text)
            except ValueError:
                pass
    if re.fullmatch(r"-?(?:(?:\d+\.\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?", text) or re.fullmatch(
        r"-?\d+[eE][+-]?\d+", text
    ):
        try:
            number = float(text)
            return number if math.isfinite(number) else None
        except ValueError:
            pass
    # Prevent spreadsheet software from interpreting source text as a formula.
    if text.startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def iter_csv(path: Path) -> tuple[list[str], Iterator[list]]:
    handle = open_text(path)
    delimiter = "\t" if ".tsv" in path.name.lower() else ","
    reader = csv.reader(handle, delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        handle.close()
        return ["empty_file"], iter(())

    def rows() -> Iterator[list]:
        try:
            for row in reader:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                elif len(row) > len(header):
                    raise ValueError(f"Row has {len(row)} fields but header has {len(header)} in {path}")
                yield [coerce_excel_value(value) for value in row]
        finally:
            handle.close()

    return [str(value) for value in header], rows()


class WorkbookBuilder:
    def __init__(self, output_path: Path, title: str, scope: str, notes: Sequence[str]):
        self.output_path = output_path
        self.title = title
        self.scope = scope
        self.notes = list(notes)
        self.workbook = Workbook(write_only=True)
        self.used_sheet_names: set[str] = set()
        self.source_records: list[dict] = []
        self._add_readme()

    def unique_name(self, requested: str) -> str:
        cleaned = re.sub(r"[\\/*?:\[\]]", "_", requested).strip("' ") or "Sheet"
        cleaned = cleaned[:31]
        candidate = cleaned
        counter = 2
        while candidate.lower() in self.used_sheet_names:
            suffix = f"_{counter}"
            candidate = f"{cleaned[:31-len(suffix)]}{suffix}"
            counter += 1
        self.used_sheet_names.add(candidate.lower())
        return candidate

    @staticmethod
    def styled_header(sheet, values: Sequence) -> list[WriteOnlyCell]:
        cells: list[WriteOnlyCell] = []
        for value in values:
            cell = WriteOnlyCell(sheet, value=str(value))
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cells.append(cell)
        return cells

    def _new_sheet(self, requested: str, header: Sequence[str]):
        name = self.unique_name(requested)
        sheet = self.workbook.create_sheet(name)
        sheet.freeze_panes = "A2"
        sheet.sheet_view.showGridLines = False
        for index, column in enumerate(header, start=1):
            width = min(34, max(12, len(str(column)) + 2))
            sheet.column_dimensions[get_column_letter(index)].width = width
        sheet.append(self.styled_header(sheet, header))
        return sheet, name

    def _add_readme(self):
        sheet_name = self.unique_name("README")
        sheet = self.workbook.create_sheet(sheet_name)
        sheet.sheet_view.showGridLines = False
        sheet.column_dimensions["A"].width = 25
        sheet.column_dimensions["B"].width = 110
        title_cell = WriteOnlyCell(sheet, value=self.title)
        title_cell.font = README_TITLE_FONT
        sheet.append([title_cell, None])
        rows = [
            ("Deliverable", self.title),
            ("Scope", self.scope),
            ("Generated", date.today().isoformat()),
            ("Packaging rule", "One workbook per Supplementary Data/Table; categories are separate worksheets."),
            ("Completeness", "All listed source-table rows are included. Oversized tables are split only at Excel's row limit."),
            ("Traceability", "See Source_Index for exact paths, SHA-256 checksums, dimensions, and transformations."),
        ]
        rows.extend(("Note", note) for note in self.notes)
        for key, value in rows:
            key_cell = WriteOnlyCell(sheet, value=key)
            key_cell.font = Font(bold=True, color="17365D")
            value_cell = WriteOnlyCell(sheet, value=value)
            value_cell.alignment = Alignment(wrap_text=True, vertical="top")
            sheet.append([key_cell, value_cell])

    def add_rows(
        self,
        requested_sheet: str,
        header: Sequence[str],
        rows: Iterable[Sequence],
        *,
        source_path: Path | None = None,
        transformation: str = "None; source table copied in full.",
        source_format: str = "generated summary",
    ) -> None:
        if len(header) > MAX_COLUMNS:
            raise ValueError(f"{requested_sheet}: {len(header)} columns exceed Excel's limit")
        part = 1
        current_rows = 0
        total_rows = 0
        sheet_names: list[str] = []
        suffix = "" if part == 1 else f"_p{part}"
        sheet, sheet_name = self._new_sheet(requested_sheet + suffix, header)
        sheet_names.append(sheet_name)
        for raw_row in rows:
            if current_rows >= MAX_DATA_ROWS:
                part += 1
                current_rows = 0
                sheet, sheet_name = self._new_sheet(f"{requested_sheet}_p{part}", header)
                sheet_names.append(sheet_name)
            converted = [coerce_excel_value(value) for value in raw_row]
            if len(converted) != len(header):
                raise ValueError(
                    f"{requested_sheet}: row has {len(converted)} cells but header has {len(header)}"
                )
            sheet.append(converted)
            current_rows += 1
            total_rows += 1
        source = source_path.resolve() if source_path else None
        self.source_records.append(
            {
                "worksheet": "; ".join(sheet_names),
                "source_file": relpath(source) if source else "Generated by packaging script",
                "source_format": source_format if source is None else (
                    "gzip-compressed CSV" if source.name.lower().endswith(".gz") else source.suffix.lstrip(".").upper()
                ),
                "rows_in_workbook": total_rows,
                "columns_in_workbook": len(header),
                "sha256": sha256(source) if source else "",
                "transformation": transformation,
            }
        )
        print(f"  {self.output_path.name}: {requested_sheet} ({total_rows:,} x {len(header)})", flush=True)

    def add_csv(self, path: Path, requested_sheet: str, transformation: str = "None; source table copied in full."):
        header, rows = iter_csv(path)
        self.add_rows(
            requested_sheet,
            header,
            rows,
            source_path=path,
            transformation=transformation,
        )

    def add_dataframe(
        self,
        frame: pd.DataFrame,
        requested_sheet: str,
        *,
        sources: Sequence[Path],
        transformation: str,
    ):
        rows = frame.itertuples(index=False, name=None)
        self.add_rows(
            requested_sheet,
            [str(c) for c in frame.columns],
            rows,
            transformation=transformation,
        )
        record = self.source_records[-1]
        record["source_file"] = "; ".join(relpath(p) for p in sources)
        record["source_format"] = "Derived from listed source tables"
        record["sha256"] = "; ".join(f"{p.name}:{sha256(p)}" for p in sources)

    def finish(self):
        index_header = [
            "worksheet",
            "source_file",
            "source_format",
            "rows_in_workbook",
            "columns_in_workbook",
            "sha256",
            "transformation",
        ]
        sheet, _ = self._new_sheet("Source_Index", index_header)
        for record in self.source_records:
            sheet.append([record[column] for column in index_header])
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.workbook.save(self.output_path)
        print(f"SAVED {relpath(self.output_path)}", flush=True)


def build_file_workbook(
    output_path: Path,
    title: str,
    scope: str,
    files: Sequence[Path],
    notes: Sequence[str],
    prefixer=None,
):
    if not files:
        raise RuntimeError(f"No source tables found for {title}")
    builder = WorkbookBuilder(output_path, title, scope, notes)
    for path in files:
        prefix = prefixer(path) if prefixer else ""
        builder.add_csv(path, short_sheet_label(path, prefix))
    builder.finish()


def marker_files(root: Path) -> list[Path]:
    fragments = (
        "state_marker_ranking_all_genes",
        "state_top25_markers_long",
        "state_top25_markers_wide",
        "gene_presence",
        "state_name_mapping",
        "state_counts",
        "state_donor_project_QC",
        "state_dataset_donor_QC",
    )
    return selected_files(root, fragments)


def build_data_s1():
    root = PROJECT / "new_result2_aging" / "08_combined_LEC_recluster" / "dataset13"
    build_file_workbook(
        DATA_DIR / "Data_S1_complete_aging_LEC_marker_results.xlsx",
        "Data S1. Complete aging LEC marker results",
        "Full state-marker ranking and its state/gene-support tables for the final aging LEC atlas.",
        marker_files(root),
        ["The all-genes ranking is not restricted to markers selected for the main figure."],
    )


def build_data_s2():
    root = PROJECT / "new_result2_aging" / "09_LEC_state_GSEA" / "dataset13"
    build_file_workbook(
        DATA_DIR / "Data_S2_complete_aging_pathway_GSEA_results.xlsx",
        "Data S2. Complete aging pathway/GSEA results",
        "All tabular outputs from the final aging LEC-state GSEA analysis.",
        tabular_files(root),
        ["The full pathway table is retained; significant and top-pathway sheets are convenience subsets produced upstream."],
    )


def build_data_s3():
    root = PROJECT / "new_result2_aging" / "11_SCENIC_analysis" / "dataset13" / "tables"
    build_file_workbook(
        DATA_DIR / "Data_S3_complete_aging_SCENIC_results.xlsx",
        "Data S3. Complete aging SCENIC results",
        "All final aging SCENIC result and robustness tables.",
        tabular_files(root),
        ["Includes complete regulon targets, AUCell summaries, RSS, balancing, and donor-bootstrap outputs."],
    )


def build_data_s4():
    root = PROJECT / "new_result2_aging" / "13_NicheNet_donor_pseudobulk" / "dataset13"
    build_file_workbook(
        DATA_DIR / "Data_S4_complete_aging_MultiNicheNet_results.xlsx",
        "Data S4. Complete aging MultiNicheNet results",
        "All tabular outputs from the final donor-pseudobulk aging MultiNicheNet analysis.",
        tabular_files(root),
        ["Includes full differential-expression, prioritization, ligand-activity, ligand-receptor, ligand-target, and QC tables."],
    )


def build_data_s5():
    root = PROJECT / "new_result2_disease" / "08_combined_LEC_recluster"
    build_file_workbook(
        DATA_DIR / "Data_S5_complete_disease_LEC_marker_results.xlsx",
        "Data S5. Complete disease LEC marker results",
        "Full state-marker ranking and its state/gene-support tables for the final disease LEC atlas.",
        marker_files(root),
        ["The all-genes ranking is not restricted to markers selected for the main figure."],
    )


def build_data_s6():
    root = PROJECT / "new_result2_disease" / "09_LEC_state_GSEA"
    build_file_workbook(
        DATA_DIR / "Data_S6_complete_disease_pathway_GSEA_results.xlsx",
        "Data S6. Complete disease pathway/GSEA results",
        "All tabular outputs from the final disease LEC-state GSEA analysis.",
        tabular_files(root),
        ["The full pathway table is retained; significant and top-pathway sheets are convenience subsets produced upstream."],
    )


def build_data_s7():
    root = PROJECT / "new_result2_disease" / "11_SCENIC_analysis" / "tables"
    build_file_workbook(
        DATA_DIR / "Data_S7_complete_disease_SCENIC_results.xlsx",
        "Data S7. Complete disease SCENIC results",
        "All final disease SCENIC result and robustness tables.",
        tabular_files(root),
        ["Includes complete regulon targets, AUCell summaries, RSS, balancing, and donor-bootstrap outputs."],
    )


def build_data_s8():
    root = PROJECT / "new_result2_disease" / "12_NicheNet_donor_pseudobulk"
    build_file_workbook(
        DATA_DIR / "Data_S8_complete_disease_MultiNicheNet_results.xlsx",
        "Data S8. Complete disease MultiNicheNet results",
        "All tabular outputs for DCM, HCM, DoxCM, and HFpEF sensitivity runs at minimum 10 and 15 LECs.",
        tabular_files(root),
        [
            "DCM, HCM, and DoxCM use the primary minimum-25-LEC disease policy.",
            "HFpEF minimum-10 and minimum-15 sensitivity runs are both retained and explicitly labeled by worksheet prefix.",
            "Tables longer than 1,048,575 data rows are split into sequential worksheets without filtering.",
        ],
        prefixer=disease_mnn_prefix,
    )




def build_data_s10():
    stage_root = PROJECT / "new_result2_spatial_disease"
    figure_root = PROJECT / "new_result2" / "disease" / "Fig06_spatial_disease_BMP_ACVR2A"
    files = tabular_files(stage_root) + tabular_files(figure_root)

    def prefix(path: Path) -> str:
        if stage_root in path.parents:
            return stage_prefix(path, stage_root)
        return "Fig06"

    build_file_workbook(
        DATA_DIR / "Data_S10_complete_disease_GeoMx_analysis_outputs.xlsx",
        "Data S10. Complete disease GeoMx analysis/output",
        "All tabular AOI QC, LEC-like AOI selection, pathway validation, and robustness outputs from the current disease GeoMx workflow.",
        files,
        [
            "Includes final Fig. 6 source tables as derivative, manuscript-facing summaries.",
            "ACVR2A assay limitations remain documented in the upstream assay-audit and detection tables; no unavailable measurement is imputed.",
        ],
        prefixer=prefix,
    )


def read_csv_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_table_s1():
    datasets_path = PROJECT / "metadata" / "datasets.csv"
    aging_path = (
        PROJECT
        / "new_result2_aging"
        / "07_LEC_description"
        / "dataset13"
        / "07_LEC_percentage_by_donor.csv"
    )
    disease_path = (
        PROJECT
        / "new_result2_disease"
        / "07_LEC_description"
        / "07_LEC_percentage_by_donor.csv"
    )
    datasets = read_csv_frame(datasets_path)
    current_ids = {
        "dataset_13_Age": "Aging LEC atlas",
        "dataset_01_DCM_HCM": "Disease LEC atlas: DCM, HCM, and NF",
        "dataset_08_DoxCMP": "Disease LEC atlas: DoxCM and NF",
        "dataset_11_HFpEF_SCP3342": "Disease LEC atlas: HFpEF and NF; threshold sensitivity",
    }
    current = datasets[datasets["dataset_id"].isin(current_ids)].copy()
    current.insert(1, "current_manuscript_role", current["dataset_id"].map(current_ids))
    publication_fields = ["paper_title", "first_author", "year"]
    current.insert(
        2,
        "publication_metadata_status",
        current[publication_fields].apply(
            lambda row: "complete_in_local_registry" if row.notna().all() else "incomplete_in_local_registry",
            axis=1,
        ),
    )
    preferred = [
        "dataset_id",
        "current_manuscript_role",
        "publication_metadata_status",
        "dataset_short_name",
        "paper_title",
        "first_author",
        "year",
        "species",
        "tissue",
        "heart_region",
        "data_type",
        "modality",
        "disease_states",
        "normal_control",
        "source_repository",
        "source_id",
        "geo_accession",
        "portal_id",
        "n_cells",
        "n_genes",
        "notes",
    ]
    current = current[[c for c in preferred if c in current.columns]]
    aging = read_csv_frame(aging_path)
    disease = read_csv_frame(disease_path)
    aging_projects = (
        aging.groupby("Project", dropna=False)
        .agg(
            n_donors=("donor_key", "nunique"),
            n_donors_with_final_LECs=("has_at_least_one_LEC", "sum"),
            n_final_LECs=("n_LEC_cells", "sum"),
        )
        .reset_index()
        .rename(columns={"Project": "project_accession"})
    )
    aging_projects["publication_metadata_status"] = "not_resolved_in_local_dataset_registry"
    builder = WorkbookBuilder(
        TABLE_DIR / "Table_S1_human_datasets_and_donor_characteristics.xlsx",
        "Table S1. Human datasets and donor characteristics",
        "Study dataset accessions/publications plus donor characteristics for the current aging and disease analyses.",
        ["The full dataset registry is retained in a separate worksheet; Current_Datasets identifies inputs represented in current final atlases."],
    )
    builder.add_dataframe(
        current,
        "Current_Datasets",
        sources=[datasets_path, aging_path, disease_path],
        transformation="Filtered registry to datasets evidenced in current final aging/disease donor mappings; selected publication/accession fields only.",
    )
    builder.add_csv(datasets_path, "Full_Dataset_Registry")
    builder.add_dataframe(
        aging_projects,
        "Aging_Project_Accessions",
        sources=[aging_path, datasets_path],
        transformation="Grouped complete aging donor QC table by Project accession; publication status flags absence of component-project citations from the local registry.",
    )
    builder.add_dataframe(
        aging,
        "Aging_Donors",
        sources=[aging_path],
        transformation="None; complete current aging donor inclusion mapping.",
    )
    builder.add_dataframe(
        disease,
        "Disease_Donors",
        sources=[disease_path],
        transformation="None; complete current disease donor inclusion mapping.",
    )
    builder.finish()


def aging_ec_counts() -> tuple[pd.DataFrame, list[Path]]:
    root = PROJECT / "new_result2_aging" / "06_EC_recluster" / "dataset13"
    paths = sorted(root.glob("Q*/dataset13_Q*_selected_EC_cells_before_reclustering.csv*"))
    frames = []
    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=["donor_id_quartile_unique", "Project", "Donor"],
            low_memory=False,
        )
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    result = (
        merged.groupby(["donor_id_quartile_unique", "Project", "Donor"], dropna=False)
        .size()
        .rename("n_EC_candidates_before_reclustering")
        .reset_index()
        .rename(columns={"donor_id_quartile_unique": "donor_key"})
    )
    return result, paths


def disease_ec_counts() -> tuple[pd.DataFrame, list[Path]]:
    root = PROJECT / "new_result2_disease" / "06_EC_recluster"
    patterns = [
        "dataset01/DCM/*selected_EC_cells_before_reclustering.csv*",
        "dataset01/HCM/*selected_EC_cells_before_reclustering.csv*",
        "dataset01/NF/*selected_EC_cells_before_reclustering.csv*",
        "dataset08/DoxCM/*selected_EC_cells_before_reclustering.csv*",
        "dataset08/NF/*selected_EC_cells_before_reclustering.csv*",
        "dataset11/HFpEF/*selected_EC_cells_before_reclustering.csv*",
        "dataset11/NF/*selected_EC_cells_before_reclustering.csv*",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(root.glob(pattern)))
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["donor_key", "dataset_folder", "condition"], low_memory=False)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    result = (
        merged.groupby(["donor_key", "dataset_folder", "condition"], dropna=False)
        .size()
        .rename("n_EC_candidates_before_reclustering")
        .reset_index()
    )
    return result, paths


def build_table_s2():
    aging_qc_path = (
        PROJECT
        / "new_result2_aging"
        / "07_LEC_description"
        / "dataset13"
        / "07_LEC_percentage_by_donor.csv"
    )
    disease_qc_path = (
        PROJECT
        / "new_result2_disease"
        / "07_LEC_description"
        / "07_LEC_percentage_by_donor.csv"
    )
    aging_map = (
        PROJECT
        / "new_result2_aging"
        / "08_7_min25_LEC_filtered_age_analysis"
        / "dataset13"
        / "08_7_all_donor_filter_mapping.csv"
    )
    disease_map = (
        PROJECT
        / "new_result2_disease"
        / "08_3_min25_LEC_filtered_analysis"
        / "tables"
        / "08_3_all_donor_filter_mapping.csv"
    )
    hfpef_min10_path = (
        PROJECT
        / "new_result2_disease"
        / "12_NicheNet_donor_pseudobulk"
        / "dataset11_HFpEF_vs_NF_min10"
        / "tables"
        / "12_selected_donors.csv"
    )
    hfpef_min15_path = (
        PROJECT
        / "new_result2_disease"
        / "12_NicheNet_donor_pseudobulk"
        / "dataset11_HFpEF_vs_NF_min15"
        / "tables"
        / "12_selected_donors.csv"
    )
    aging_qc = read_csv_frame(aging_qc_path)
    disease_qc = read_csv_frame(disease_qc_path)
    aging_filter = read_csv_frame(aging_map)
    disease_filter = read_csv_frame(disease_map)
    hfpef_min10 = read_csv_frame(hfpef_min10_path)
    hfpef_min15 = read_csv_frame(hfpef_min15_path)

    aging_summary = aging_qc.merge(
        aging_filter[["donor_key", "n_total_LEC_cells", "retained_min25_LECs"]],
        on="donor_key",
        how="left",
        validate="one_to_one",
    )
    matched_aging = aging_summary["n_total_LEC_cells"].notna()
    if not (
        aging_summary.loc[matched_aging, "n_LEC_cells"].astype(int)
        == aging_summary.loc[matched_aging, "n_total_LEC_cells"].astype(int)
    ).all():
        raise RuntimeError("Aging Step 07 and final donor-map LEC counts disagree")
    aging_summary.insert(0, "analysis", "aging")
    aging_summary["n_final_LECs"] = aging_summary["n_LEC_cells"]
    aging_summary["minimum_LECs"] = 25
    aging_summary["retained_minimum_LECs"] = (
        aging_summary["retained_min25_LECs"].astype("boolean").fillna(False).astype(bool)
    )
    aging_summary["filter_status"] = aging_summary["retained_minimum_LECs"].map(
        {True: "retained_min25", False: "excluded_below25_or_zero_LECs"}
    )

    disease_summary = disease_qc.merge(
        disease_filter[
            ["donor_key", "n_final_LECs", "minimum_LECs", "retained_minimum_LECs", "filter_status"]
        ],
        on="donor_key",
        how="left",
        validate="one_to_one",
        suffixes=("", "_final_map"),
    )
    matched_disease = disease_summary["n_final_LECs"].notna()
    if not (
        disease_summary.loc[matched_disease, "n_LEC_cells"].astype(int)
        == disease_summary.loc[matched_disease, "n_final_LECs"].astype(int)
    ).all():
        raise RuntimeError("Disease Step 07 and final donor-map LEC counts disagree")
    disease_summary.insert(0, "analysis", "disease")
    disease_summary["n_final_LECs"] = disease_summary["n_final_LECs"].fillna(disease_summary["n_LEC_cells"])
    disease_summary["minimum_LECs"] = disease_summary["minimum_LECs"].fillna(25).astype(int)
    disease_summary["retained_minimum_LECs"] = (
        disease_summary["retained_minimum_LECs"].astype("boolean").fillna(False).astype(bool)
    )
    disease_summary["filter_status"] = disease_summary["filter_status"].fillna("excluded_zero_final_LECs")
    disease_summary["included_in_HFpEF_MNN_min10"] = disease_summary["donor_key"].isin(
        set(hfpef_min10["analysis_donor_key"])
    )
    disease_summary["included_in_HFpEF_MNN_min15"] = disease_summary["donor_key"].isin(
        set(hfpef_min15["analysis_donor_key"])
    )
    keep = [
        "analysis",
        "donor_key",
        "Project",
        "dataset_folder",
        "condition",
        "age_group",
        "age_numeric",
        "age",
        "Sex",
        "sex",
        "n_all_QC_cells",
        "n_EC_cells",
        "n_LEC_cells",
        "n_final_LECs",
        "minimum_LECs",
        "retained_minimum_LECs",
        "filter_status",
        "included_in_HFpEF_MNN_min10",
        "included_in_HFpEF_MNN_min15",
    ]
    combined = pd.concat(
        [aging_summary.reindex(columns=keep), disease_summary.reindex(columns=keep)],
        ignore_index=True,
    )
    combined = combined.dropna(axis=1, how="all")
    sources = [aging_qc_path, disease_qc_path, aging_map, disease_map, hfpef_min10_path, hfpef_min15_path]
    builder = WorkbookBuilder(
        TABLE_DIR / "Table_S2_donor_QC_EC_LEC_numbers_and_inclusion.xlsx",
        "Table S2. Donor-level QC, EC/LEC numbers, and inclusion/exclusion",
        "Donor-level selected-EC counts, final LEC counts, and minimum-LEC inclusion status for current aging and disease analyses.",
        ["All-QC-cell, EC, and LEC counts come directly from Step 07 donor tables; primary minimum-25 inclusion flags come from final donor-filter mappings. Zero-LEC donors absent from the latter are restored as excluded. Separate columns record exact donor membership in HFpEF minimum-10 and minimum-15 MultiNicheNet sensitivity runs."],
    )
    builder.add_dataframe(
        combined,
        "Donor_QC_Summary",
        sources=sources,
        transformation="Left-joined authoritative Step 07 donor QC/count tables to final minimum-25 donor mappings; restored zero-LEC donors as excluded; no donor omitted.",
    )
    builder.add_csv(aging_qc_path, "Aging_Donor_QC_Source")
    builder.add_csv(disease_qc_path, "Disease_Donor_QC_Source")
    builder.add_csv(aging_map, "Aging_Inclusion_Source")
    builder.add_csv(disease_map, "Disease_Inclusion_Source")
    builder.add_csv(hfpef_min10_path, "HFpEF_MNN_min10_Donors")
    builder.add_csv(hfpef_min15_path, "HFpEF_MNN_min15_Donors")
    builder.finish()


def state_definition_frame(source_path: Path) -> pd.DataFrame:
    data = pd.read_csv(source_path, low_memory=False)
    definitions = (
        data[["defining_state", "state_definition"]]
        .dropna()
        .drop_duplicates()
        .sort_values("defining_state")
        .rename(columns={"defining_state": "state", "state_definition": "proposed_definition"})
    )
    markers = (
        data.loc[data["record_type"].eq("marker_expression"), ["defining_state", "gene", "marker_order"]]
        .dropna(subset=["defining_state", "gene"])
        .drop_duplicates()
        .sort_values(["defining_state", "marker_order"])
        .groupby("defining_state")["gene"]
        .apply(lambda values: ", ".join(values.astype(str)))
        .rename("representative_marker_genes")
        .reset_index()
        .rename(columns={"defining_state": "state"})
    )
    result = definitions.merge(markers, on="state", how="left")
    result["state"] = result["state"].astype(int)
    result["representative_marker_genes"] = result["representative_marker_genes"].fillna(
        "No robust major-marker set selected for the main figure"
    )
    return result


def build_state_table(number: int, label: str, source_path: Path, output_name: str):
    frame = state_definition_frame(source_path)
    builder = WorkbookBuilder(
        TABLE_DIR / output_name,
        f"Table S{number}. Final {label} LEC-state definitions",
        f"Final proposed definitions and representative marker genes used in the current {label} marker/pathway panel.",
        ["State labels and marker order are derived from the current figure source data; no new biological labels were introduced."],
    )
    builder.add_dataframe(
        frame,
        "State_Definitions",
        sources=[source_path],
        transformation="Deduplicated state-definition records and concatenated representative markers in the recorded marker order.",
    )
    builder.add_csv(source_path, "Figure_Source_Data")
    builder.finish()


def build_table_s3():
    build_state_table(
        3,
        "aging",
        PROJECT / "new_result2" / "aging" / "Fig02C_aging_marker_GSEA_map" / "Fig02C_source_data.csv",
        "Table_S3_final_aging_LEC_state_definitions.xlsx",
    )


def build_table_s4():
    build_state_table(
        4,
        "disease",
        PROJECT / "new_result2" / "disease" / "Fig04C_disease_marker_GSEA_map" / "Fig04C_source_data.csv",
        "Table_S4_final_disease_LEC_state_definitions.xlsx",
    )


def build_table_s5():
    files = [
        (
            PROJECT / "new_result2_spatial_disease" / "13_1_spatial_disease_QC" / "13_1_design_summary.csv",
            "Disease_Design",
        ),
        (
            PROJECT / "new_result2_spatial_disease" / "13_1_spatial_disease_QC" / "13_1_AOI_metadata_and_QC.csv",
            "Disease_AOIs",
        ),
        (
            PROJECT / "new_result2_spatial_disease" / "13_2_disease_LEC_reference" / "13_2_selected_AOI_LEC_evidence.csv",
            "Disease_Selected_AOIs",
        ),
        (
            PROJECT / "new_result2_spatial_disease" / "13_2_disease_LEC_reference" / "13_2_selected_donor_LEC_evidence.csv",
            "Disease_Selected_Donors",
        ),
    ]
    builder = WorkbookBuilder(
        TABLE_DIR / "Table_S5_spatial_datasets_donors_sections_AOIs_and_inclusion.xlsx",
        "Table S5. Spatial datasets, donors, sections/AOIs, and inclusion",
        "Disease GeoMx donor/AOI design, QC, selection, and inclusion information.",
        ["The disease sheets retain all AOIs and the selected LEC-like AOI/donor summaries so exclusions remain auditable."],
    )
    for path, sheet in files:
        if not path.exists():
            raise FileNotFoundError(path)
        builder.add_csv(path, sheet)
    builder.finish()


def flatten_json(value, prefix: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_json(child, child_prefix)
    elif isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            yield prefix, "; ".join(str(item) for item in value)
        else:
            for index, child in enumerate(value):
                yield from flatten_json(child, f"{prefix}[{index}]")
    else:
        yield prefix, "" if value is None else str(value)


def config_files() -> list[Path]:
    explicit = [
        PROJECT / "new_result2_aging" / "08_combined_LEC_recluster" / "dataset13" / "08_run_config.json",
        PROJECT / "new_result2_aging" / "09_LEC_state_GSEA" / "dataset13" / "09_run_config.json",
        PROJECT / "new_result2_aging" / "11_SCENIC_analysis" / "dataset13" / "11_run_config.json",
        PROJECT / "new_result2_aging" / "13_NicheNet_donor_pseudobulk" / "dataset13" / "13_run_config.json",
        PROJECT / "new_result2_disease" / "08_combined_LEC_recluster" / "08_run_config.json",
        PROJECT / "new_result2_disease" / "09_LEC_state_GSEA" / "09_run_config.json",
        PROJECT / "new_result2_disease" / "11_SCENIC_analysis" / "11_run_config.json",
        PROJECT / "new_result2_disease" / "12_NicheNet_donor_pseudobulk" / "12_run_config.json",
    ]
    explicit.extend(sorted((PROJECT / "new_result2_spatial_disease").glob("*/13_*_run_config.json")))
    explicit.extend(sorted((PROJECT / "new_result2_disease" / "12_NicheNet_donor_pseudobulk").glob("*/12_run_config.json")))
    return [path for path in explicit if path.exists()]


def build_table_s6():
    configs = config_files()
    parameter_rows = []
    for path in configs:
        with path.open("r", encoding="utf-8-sig") as handle:
            config = json.load(handle)
        analysis = path.parent.name
        for parameter, value in flatten_json(config):
            parameter_rows.append(
                {
                    "analysis": analysis,
                    "parameter": parameter,
                    "value": value,
                    "source_file": relpath(path),
                }
            )
    parameters = pd.DataFrame(parameter_rows)

    packages = []
    for package in [
        "anndata",
        "h5py",
        "matplotlib",
        "numpy",
        "openpyxl",
        "pandas",
        "scanpy",
        "scikit-learn",
        "scipy",
        "seaborn",
        "statsmodels",
    ]:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed in packaging environment"
        packages.append(
            {
                "environment": "current Python analysis/packaging environment",
                "software_or_package": package,
                "version": version,
                "evidence": "queried with importlib.metadata while building Table S6",
            }
        )
    packages.insert(
        0,
        {
            "environment": "current Python analysis/packaging environment",
            "software_or_package": "Python",
            "version": platform.python_version(),
            "evidence": sys.executable,
        },
    )
    package_frame = pd.DataFrame(packages)

    session_paths = [
        PROJECT / "new_result2_aging" / "13_NicheNet_donor_pseudobulk" / "dataset13" / "13_sessionInfo.txt",
        PROJECT / "new_result2_disease" / "12_NicheNet_donor_pseudobulk" / "12_sessionInfo.txt",
    ]
    builder = WorkbookBuilder(
        TABLE_DIR / "Table_S6_software_versions_and_major_analysis_parameters.xlsx",
        "Table S6. Software/packages, versions, and major analysis parameters",
        "Saved run-configuration parameters, current Python package versions, and saved R session information.",
        ["Saved run configs and R sessionInfo are authoritative for their corresponding analyses; current Python versions are explicitly labeled as the build environment."],
    )
    builder.add_dataframe(
        parameters,
        "Major_Parameters",
        sources=configs,
        transformation="Flattened nested JSON keys to one parameter per row; values otherwise unchanged.",
    )
    builder.add_dataframe(
        package_frame,
        "Python_Packages",
        sources=[],
        transformation="Versions queried from the active Python environment at workbook-build time.",
    )
    for label, path in zip(("R_Session_Aging", "R_Session_Disease"), session_paths):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        builder.add_rows(
            label,
            ["line_number", "session_info"],
            ((index, line) for index, line in enumerate(lines, start=1)),
            source_path=path,
            transformation="Saved sessionInfo text represented as one numbered line per row.",
            source_format="TXT",
        )
    builder.finish()


BUILDERS = {
    "Data_S1": build_data_s1,
    "Data_S2": build_data_s2,
    "Data_S3": build_data_s3,
    "Data_S4": build_data_s4,
    "Data_S5": build_data_s5,
    "Data_S6": build_data_s6,
    "Data_S7": build_data_s7,
    "Data_S8": build_data_s8,
    "Data_S10": build_data_s10,
    "Table_S1": build_table_s1,
    "Table_S2": build_table_s2,
    "Table_S3": build_table_s3,
    "Table_S4": build_table_s4,
    "Table_S5": build_table_s5,
    "Table_S6": build_table_s6,
}


def validate_workbooks(keys: Sequence[str]) -> None:
    failures = []
    for key in keys:
        folder = DATA_DIR if key.startswith("Data_") else TABLE_DIR
        matches = sorted(folder.glob(f"{key}_*.xlsx"))
        if len(matches) != 1:
            failures.append(f"{key}: expected exactly one workbook, found {len(matches)}")
            continue
        path = matches[0]
        workbook = load_workbook(path, read_only=True, data_only=False)
        names = workbook.sheetnames
        if "README" not in names or "Source_Index" not in names:
            failures.append(f"{key}: missing README or Source_Index")
        source_rows = (
            max(0, sum(1 for _ in workbook["Source_Index"].iter_rows(values_only=True)) - 1)
            if "Source_Index" in names
            else 0
        )
        workbook.close()
        if source_rows < 1:
            failures.append(f"{key}: empty Source_Index")
        print(f"VALID {path.name}: {len(names)} sheets, {path.stat().st_size / 1048576:.1f} MB", flush=True)
    if failures:
        raise RuntimeError("Workbook validation failed:\n" + "\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(BUILDERS),
        help="Build only selected deliverables (default: all).",
    )
    parser.add_argument("--validate-only", action="store_true", help="Validate existing workbooks without rebuilding.")
    args = parser.parse_args()
    keys = args.only or list(BUILDERS)
    if not args.validate_only:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        for key in keys:
            print(f"BUILD {key}", flush=True)
            BUILDERS[key]()
    validate_workbooks(keys)


if __name__ == "__main__":
    main()
