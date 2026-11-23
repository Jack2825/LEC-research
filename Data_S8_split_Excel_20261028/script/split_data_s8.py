"""Lossless packaging of existing Data S8 worksheets into smaller Excel files.

No cell XML is reserialized, no tables are filtered, and no analysis is rerun.
The original workbook is opened read-only. Existing output directories are refused.
Run from the project root with .venv/Scripts/python.exe -B -X utf8 <this script>.
Dependency: openpyxl (for the small index and read-only workbook verification).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import posixpath
import re
import sys
import zipfile
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "Data_S8_split_Excel_20261028/inputs/Data_S8_complete_disease_MultiNicheNet_results.xlsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "result"
SS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"s": SS}
META = ("README", "Source_Index")
NEW_SHEET_PATH = "xl/worksheets/split_readme.xml"
CHUNK = 4 * 1024 * 1024


def progress(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def xml_bytes(element, namespace):
    ET.register_namespace("", namespace)
    ET.register_namespace("r", REL)
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def sheet_map(archive):
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {}
    for rel in rels:
        target = rel.get("Target")
        targets[rel.get("Id")] = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    return OrderedDict(
        (sheet.get("name"), targets[sheet.get(f"{{{REL}}}id")])
        for sheet in workbook.find("s:sheets", NS)
    )


def small_table(xml):
    rows = []
    for row in ET.fromstring(xml).findall("s:sheetData/s:row", NS):
        values = []
        for cell in row.findall("s:c", NS):
            if cell.get("t") == "inlineStr":
                values.append("".join(t.text or "" for t in cell.findall(".//s:t", NS)))
            else:
                values.append(cell.findtext("s:v", default="", namespaces=NS))
        rows.append(values)
    return [dict(zip(rows[0], row)) for row in rows[1:]]


def category(name):
    cohort = name.split("_", 1)[0]
    if cohort in ("Overview", "HFpEF10", "HFpEF15"):
        return cohort, "all_tables"
    if "_MNN_ligand_activity" in name:
        return cohort, "ligand_activity"
    if "_MNN_sample_priority" in name:
        return cohort, "sample_priority"
    if "_MNN_normPB_" in name:
        return cohort, "expression"
    return cohort, "results_QC_metadata"


def create_plan(archive, mapping, maximum):
    groups = OrderedDict()
    for name, path in mapping.items():
        if name not in META:
            groups.setdefault(category(name), []).append(name)
    # Leave >1 MB below the requested cap for duplicated metadata/package overhead.
    payload_limit = maximum - 1_000_000
    plan = []
    for (cohort, label), names in groups.items():
        batches, current, size = [], [], 0
        for name in names:
            compressed = archive.getinfo(mapping[name]).compress_size
            if compressed > payload_limit:
                raise ValueError(f"An intact worksheet exceeds the requested cap: {name}")
            if current and size + compressed > payload_limit:
                batches.append(current)
                current, size = [], 0
            current.append(name)
            size += compressed
        if current:
            batches.append(current)
        for i, batch in enumerate(batches, 1):
            suffix = f"_part{i}" if len(batches) > 1 else ""
            number = len(plan) + 1
            plan.append({
                "file": f"Data_S8_{number:02d}_{cohort}_{label}{suffix}.xlsx",
                "cohort": cohort,
                "category": label,
                "sheets": batch,
                "estimated_payload_bytes": sum(archive.getinfo(mapping[n]).compress_size for n in batch),
            })
    return plan


def split_readme(part, total, source, digest, max_bytes):
    rows = [
        ["Data S8 — split Excel release", part["file"]],
        ["Contents", f"{part['cohort']}: {part['category']}; one of {total} data workbooks."],
        ["Packaging", f"Existing worksheets distributed among files smaller than {max_bytes / 1e6:g} MB; no filtering or new analysis."],
        ["Data preservation", "All original worksheet XML is copied byte-for-byte, retaining values, precision, types, headers, row order and formatting."],
        ["Original source workbook", source.name],
        ["Original workbook SHA-256", digest],
        ["Complete release index", "Data_S8_split_index.xlsx lists every worksheet and its destination file."],
        ["Original README / Source_Index", "These two sheets are unchanged historical metadata for the complete source workbook, NOT a contents list for this individual part."],
        ["Historical packaging statement", "The original README describes the previous one-workbook package. This split release uses multiple workbooks."],
        ["Existing _p2 sheet suffix", "An original continuation of a table at Excel's row limit; all original parts are retained. Locate them in the split index."],
        ["Units", "File sizes use decimal MB = 1,000,000 bytes."],
        ["Original data worksheets in this file", ""],
    ] + [[name, "Unchanged original worksheet"] for name in part["sheets"]]
    book = Workbook()
    ws = book.active
    ws.title = "Split_Readme"
    for row in rows:
        ws.append(row)
    ws.column_dimensions["A"].width = 44
    ws.column_dimensions["B"].width = 110
    ws.freeze_panes = "A2"
    memory = io.BytesIO()
    book.save(memory)
    # This new sheet deliberately has no styled cells: style 0 exists in the source.
    with zipfile.ZipFile(memory) as archive:
        return archive.read("xl/worksheets/sheet1.xml")


def scan_stream(source, destination=None):
    """Hash exact XML, count rows, and reject formulas without materializing cells."""
    digest = hashlib.sha256()
    row_count = formula_count = 0
    tail = b""
    for block in iter(lambda: source.read(CHUNK), b""):
        digest.update(block)
        if destination is not None:
            destination.write(block)
        combined = tail + block
        # Count only tokens completed in the newly read block (including boundary tokens).
        row_count += sum(m.end() > len(tail) for m in re.finditer(rb"<row(?:\s|>|/)", combined))
        formula_count += sum(m.end() > len(tail) for m in re.finditer(rb"<f(?:\s|>|/)", combined))
        tail = combined[-16:]
    return {"sha256": digest.hexdigest(), "xml_rows": row_count, "formula_cells": formula_count}


def write_part(source_zip, mapping, part, output, readme, source_stats):
    kept = set(part["sheets"]) | set(META)
    kept_paths = {mapping[n] for n in kept}
    excluded_paths = set(mapping.values()) - kept_paths
    workbook = ET.fromstring(source_zip.read("xl/workbook.xml"))
    sheets = workbook.find("s:sheets", NS)
    original = {s.get("name"): copy.deepcopy(s) for s in sheets}
    sheets.clear()
    sheets.append(ET.Element(f"{{{SS}}}sheet", {
        "name": "Split_Readme", "sheetId": str(len(mapping) + 1),
        "state": "visible", f"{{{REL}}}id": "rIdSplitReadme",
    }))
    for name in part["sheets"] + list(META):
        sheets.append(original[name])
    for view in workbook.findall("s:bookViews/s:workbookView", NS):
        view.set("activeTab", "0")
        view.set("firstSheet", "0")
    relationships = ET.fromstring(source_zip.read("xl/_rels/workbook.xml.rels"))
    retained_ids = {s.get(f"{{{REL}}}id") for s in sheets}
    for rel in list(relationships):
        if rel.get("Type") == REL + "/worksheet" and rel.get("Id") not in retained_ids:
            relationships.remove(rel)
    relationships.append(ET.Element(f"{{{PKG}}}Relationship", {
        "Type": REL + "/worksheet", "Target": "/" + NEW_SHEET_PATH, "Id": "rIdSplitReadme",
    }))
    content_types = ET.fromstring(source_zip.read("[Content_Types].xml"))
    for item in list(content_types):
        if item.get("PartName", "").lstrip("/") in excluded_paths:
            content_types.remove(item)
    content_types.append(ET.Element(f"{{{CT}}}Override", {
        "PartName": "/" + NEW_SHEET_PATH,
        "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml",
    }))
    replacements = {
        "xl/workbook.xml": xml_bytes(workbook, SS),
        "xl/_rels/workbook.xml.rels": xml_bytes(relationships, PKG),
        "[Content_Types].xml": xml_bytes(content_types, CT),
    }
    reverse = {path: name for name, path in mapping.items()}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as target_zip:
        for info in source_zip.infolist():
            if info.filename in excluded_paths:
                continue
            if info.filename in replacements:
                target_zip.writestr(info.filename, replacements[info.filename])
            elif info.filename in kept_paths:
                with source_zip.open(info.filename) as inp, target_zip.open(info.filename, "w", force_zip64=True) as out:
                    stats = scan_stream(inp, out)
                name = reverse[info.filename]
                if stats["formula_cells"]:
                    raise ValueError(f"Formula cells detected; cross-workbook dependencies require review: {name}")
                if name in source_stats:
                    assert source_stats[name] == stats
                source_stats[name] = stats
            else:
                target_zip.writestr(info.filename, source_zip.read(info.filename))
        target_zip.writestr(NEW_SHEET_PATH, readme)


def write_csv(path, records):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def index_book(output, file_records, sheet_records, source_records, notes):
    book = Workbook()
    book.remove(book.active)
    for title, records in [("README", notes), ("Files", file_records), ("Worksheet_map", sheet_records), ("Original_sources", source_records)]:
        ws = book.create_sheet(title)
        ws.append(list(records[0]))
        for record in records:
            ws.append(list(record.values()))
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="17365D")
        for column in ws.columns:
            width = min(85, max(15, max(len(str(c.value or "")) for c in column) + 2))
            ws.column_dimensions[column[0].column_letter].width = width
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top")
    for row in book["Files"].iter_rows(min_row=2):
        row[0].hyperlink = row[0].value
        row[0].style = "Hyperlink"
    book.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-mb", type=float, default=50.0, help="Maximum decimal MB per data workbook (default 50)")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    maximum = int(args.max_mb * 1_000_000)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {output}")
    source_before = source.stat()
    original_hash = file_hash(source)
    progress(f"Source: {source.name}; {source_before.st_size / 1e6:.3f} MB; SHA-256 {original_hash}")
    with zipfile.ZipFile(source) as archive:
        mapping = sheet_map(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        defined = workbook.find("s:definedNames", NS)
        if defined is not None and len(defined):
            raise ValueError("Defined names require dependency review before splitting.")
        allowed = set(mapping.values()) | {
            "docProps/app.xml", "docProps/core.xml", "xl/theme/theme1.xml", "xl/styles.xml",
            "_rels/.rels", "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "[Content_Types].xml",
        }
        unexpected = set(archive.namelist()) - allowed
        if unexpected:
            raise ValueError(f"Review additional package dependencies before splitting: {unexpected}")
        source_records = small_table(archive.read(mapping["Source_Index"]))
        source_by_sheet = {}
        for record in source_records:
            for name in record["worksheet"].split("; "):
                if name in source_by_sheet:
                    raise ValueError(f"Duplicate worksheet in source index: {name}")
                source_by_sheet[name] = record
        assert set(source_by_sheet) == set(mapping) - set(META)
        plan = create_plan(archive, mapping, maximum)
        coverage = Counter(name for part in plan for name in part["sheets"])
        assert set(coverage) == set(source_by_sheet) and set(coverage.values()) == {1}
        output.mkdir(parents=True, exist_ok=False)
        (output / "split_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        progress(f"Plan: {len(plan)} workbooks; {len(coverage)} data worksheets; original metadata repeated in each file.")
        stats, files, worksheet_records = {}, [], []
        for number, part in enumerate(plan, 1):
            progress(f"Writing {number}/{len(plan)}: {part['file']}")
            target = output / part["file"]
            readme = split_readme(part, len(plan), source, original_hash, maximum)
            write_part(archive, mapping, part, target, readme, stats)
            assert target.stat().st_size < maximum, f"Size cap exceeded: {target}"
            # Read every output entry fully: ZIP CRC verification, plus exact XML hashes.
            with zipfile.ZipFile(target) as check:
                target_map = sheet_map(check)
                assert list(target_map) == ["Split_Readme"] + part["sheets"] + list(META)
                for name, path in target_map.items():
                    with check.open(path) as stream:
                        checked = scan_stream(stream)
                    if name != "Split_Readme":
                        assert checked == stats[name], f"Worksheet mismatch: {name}"
                for entry in check.namelist():
                    if entry not in target_map.values():
                        content = check.read(entry)
                        if entry not in {"xl/workbook.xml", "xl/_rels/workbook.xml.rels", "[Content_Types].xml"}:
                            assert content == archive.read(entry), f"Shared resource mismatch: {entry}"
            excel = load_workbook(target, read_only=True, data_only=False, keep_links=False)
            assert excel.sheetnames == ["Split_Readme"] + part["sheets"] + list(META)
            for ws in excel:
                assert next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None) is not None
            excel.close()
            row_total = 0
            for name in part["sheets"]:
                entry = source_by_sheet[name]
                rows = stats[name]["xml_rows"] - 1
                row_total += rows
                worksheet_records.append({
                    "original_worksheet": name, "split_file": part["file"],
                    "data_rows": rows, "columns": int(entry["columns_in_workbook"]),
                    "worksheet_xml_sha256": stats[name]["sha256"],
                    "source_table": entry["source_file"],
                    "original_table_worksheets": entry["worksheet"],
                    "source_table_sha256_as_recorded": entry["sha256"],
                    "preservation": "Exact worksheet XML SHA-256 match",
                })
            files.append({
                "file": part["file"], "cohort": part["cohort"], "category": part["category"],
                "data_worksheets": len(part["sheets"]), "data_rows": row_total,
                "bytes": target.stat().st_size, "MB": round(target.stat().st_size / 1e6, 3),
                "sha256": file_hash(target),
            })
            progress(f"Verified {part['file']}: {files[-1]['MB']:.3f} MB; {row_total:,} data rows; all worksheet hashes match.")
        for entry in source_records:
            names = entry["worksheet"].split("; ")
            assert sum(stats[n]["xml_rows"] - 1 for n in names) == int(entry["rows_in_workbook"]), entry["worksheet"]
        assert file_hash(source) == original_hash, "Original workbook changed during packaging"
        after = source.stat()
        assert (after.st_size, after.st_mtime_ns) == (source_before.st_size, source_before.st_mtime_ns)
        total_rows = sum(f["data_rows"] for f in files)
        for metadata in META:
            worksheet_records.append({
                "original_worksheet": metadata, "split_file": "Repeated unchanged in every data workbook",
                "data_rows": "", "columns": "", "worksheet_xml_sha256": stats[metadata]["sha256"],
                "source_table": "Historical workbook metadata", "original_table_worksheets": metadata,
                "source_table_sha256_as_recorded": "", "preservation": "Exact worksheet XML SHA-256 match",
            })
        notes = [{"item": k, "description": v} for k, v in [
            ("Deliverable", "Data S8. Complete disease MultiNicheNet results — split Excel release"),
            ("Original", str(source)), ("Original SHA-256", original_hash),
            ("Scope", f"{len(source_records)} original tables, {len(coverage)} data worksheets, {total_rows:,} data rows; all retained."),
            ("Files", f"{len(files)} data workbooks, each below {maximum / 1e6:g} decimal MB; this small index is additional."),
            ("Preservation", "Every original worksheet XML matches its output SHA-256 exactly. No data cells, formulas, styles or precision were changed; no new analysis."),
            ("Metadata", "Original README and Source_Index are repeated unchanged as historical metadata; each part opens on a new Split_Readme."),
            ("Historical README", "The original one-workbook packaging description is superseded by this split release, not changed in the preserved original README."),
            ("Continuation tables", "Existing _p2 worksheets remain unchanged. Worksheet_map identifies every component of each original table."),
            ("Grouping", "Grouped by existing dataset prefix and table type. Groups above the cap are split in original worksheet order. No result-based selection."),
            ("Upload", "Use these data workbooks together with this index. Do not upload only one part as the complete Data S8."),
            ("Size meaning", "Splitting lowers individual file sizes; combined storage remains approximately the original size. A total upload limit is not reduced."),
        ]]
        write_csv(output / "Data_S8_split_files.csv", files)
        write_csv(output / "Data_S8_split_worksheet_map.csv", worksheet_records)
        index_book(output / "Data_S8_split_index.xlsx", files, worksheet_records, source_records, notes)
        verification = {
            "status": "PASS", "source": str(source), "source_sha256_before_and_after": original_hash,
            "source_size_bytes": source_before.st_size, "source_mtime_unchanged": True,
            "original_worksheets": len(mapping), "data_worksheets": len(coverage),
            "original_source_tables": len(source_records), "data_rows": total_rows,
            "all_data_worksheets_present_exactly_once": True,
            "original_metadata_repeated_and_unchanged": list(META),
            "all_original_worksheet_xml_hashes_match": True,
            "all_shared_styles_theme_and_document_properties_unchanged": True,
            "all_output_zip_entries_crc_checked": True, "all_workbooks_openpyxl_readable": True,
            "all_source_table_row_totals_match_index": True, "formula_cells": sum(s["formula_cells"] for s in stats.values()),
            "maximum_allowed_bytes_per_data_workbook": maximum,
            "data_workbooks": len(files), "total_data_workbook_bytes": sum(f["bytes"] for f in files),
            "smallest_data_workbook_bytes": min(f["bytes"] for f in files),
            "largest_data_workbook_bytes": max(f["bytes"] for f in files),
            "python_version": sys.version, "files": files,
        }
        (output / "validation.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
        lines = ["# Data S8 — split Excel release", "", "Start with **Data_S8_split_index.xlsx**.", ""]
        lines += [f"- **{r['item']}:** {r['description']}" for r in notes]
        lines += ["", "All integrity checks passed; see validation.json.", "", "| File | MB | Data worksheets | Data rows |", "|---|---:|---:|---:|"]
        lines += [f"| {f['file']} | {f['MB']:.3f} | {f['data_worksheets']} | {f['data_rows']:,} |" for f in files]
        (output / "START_HERE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        progress(f"COMPLETE: {len(files)} data workbooks; {total_rows:,} data rows preserved; original unchanged. Output: {output}")


if __name__ == "__main__":
    main()
