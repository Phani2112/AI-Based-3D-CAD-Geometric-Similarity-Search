import argparse
import csv
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
OUTPUT_HEADERS = [
    "query_file",
    "query_uuid",
    "similia_rank",
    "uuid",
    "similia_similarity",
    "cadgcl_rank",
    "cadgcl_score",
    "appears_in_cadgcl_top_k",
]


def _column_index(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def read_xlsx_rows(path):
    with zipfile.ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                text = "".join(node.text or "" for node in item.findall(".//a:t", NS))
                shared_strings.append(text)

        worksheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in worksheet.findall(".//a:row", NS):
            values = []
            for cell in row.findall("a:c", NS):
                cell_index = _column_index(cell.get("r", "A1"))
                while len(values) < cell_index:
                    values.append("")
                value_node = cell.find("a:v", NS)
                value = "" if value_node is None else value_node.text or ""
                if cell.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//a:t", NS))
                values.append(value)
            rows.append(values)
        return rows


def write_xlsx_rows(path, headers, rows):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = [headers] + [[row.get(header, "") for header in headers] for row in rows]

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml())
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml())
        archive.writestr("xl/worksheets/sheet1.xml", _worksheet_xml(table))
        archive.writestr("xl/styles.xml", _styles_xml())


def _content_types_xml():
    return """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""


def _root_rels_xml():
    return """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""


def _workbook_xml():
    return """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="comparison" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""


def _workbook_rels_xml():
    return """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _styles_xml():
    return """<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="1"><fill><patternFill patternType="none"/></fill></fills>
<borders count="1"><border/></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>"""


def _cell_ref(row_index, column_index):
    letters = ""
    number = column_index + 1
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"{letters}{row_index}"


def _worksheet_xml(table):
    rows_xml = []
    for row_index, row in enumerate(table, 1):
        cells = []
        for column_index, value in enumerate(row):
            ref = _cell_ref(row_index, column_index)
            text = "" if value is None else str(value)
            escaped = _escape_xml(text)
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escaped}</t></is></c>')
        rows_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>{''.join(rows_xml)}</sheetData>
</worksheet>"""


def _escape_xml(value):
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _normalize_uuid(value):
    return Path(str(value).strip()).stem


def _to_float(value):
    if value == "" or value is None:
        return ""
    return float(str(value).replace(",", "."))


def load_similia_rows(meta_dir):
    rows = []
    for path in sorted(Path(meta_dir).glob("[0-9][0-9][0-9].xlsx")):
        table = read_xlsx_rows(path)
        if len(table) <= 1:
            continue
        query_uuid = _normalize_uuid(table[1][0])
        for rank, row in enumerate(table[1:], 1):
            if not row or not str(row[0]).strip():
                continue
            rows.append(
                {
                    "query_file": path.name,
                    "query_uuid": query_uuid,
                    "similia_rank": rank,
                    "uuid": _normalize_uuid(row[0]),
                    "similia_similarity": _to_float(row[1] if len(row) > 1 else ""),
                }
            )
    return rows


def annotate_with_cadgcl(rows, search_results):
    annotated = []
    for row in rows:
        result_lookup = {
            uuid: {"rank": rank, "score": score}
            for rank, (uuid, score) in enumerate(search_results.get(row["query_uuid"], []), 1)
        }
        match = result_lookup.get(row["uuid"])
        updated = dict(row)
        if match:
            updated["cadgcl_rank"] = match["rank"]
            updated["cadgcl_score"] = match["score"]
            updated["appears_in_cadgcl_top_k"] = True
        else:
            updated["cadgcl_rank"] = ""
            updated["cadgcl_score"] = ""
            updated["appears_in_cadgcl_top_k"] = False
        annotated.append(updated)
    return annotated


def write_csv_rows(path, headers, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def build_uuid_index(metadata):
    return {item["uuid"]: index for index, item in enumerate(metadata)}


def compute_cadgcl_search_results(dataset, query_uuids, k):
    from src.data.dataset_registry import dataset_artifacts
    from src.search.predictor import CADGCLPredictor

    artifacts = dataset_artifacts(dataset)
    metadata = torch.load(artifacts.metadata_path, weights_only=False)
    uuid_to_id = build_uuid_index(metadata)
    id_to_uuid = {index: item["uuid"] for index, item in enumerate(metadata)}

    predictor = CADGCLPredictor(artifacts.model_path, artifacts.embeddings_path)
    predictor.load()

    results = {}
    for query_uuid in sorted(set(query_uuids)):
        if query_uuid not in uuid_to_id:
            results[query_uuid] = []
            continue
        query_id = uuid_to_id[query_uuid]
        scored = predictor.search_with_scores(query_id, k=k)
        results[query_uuid] = [(id_to_uuid[model_id], score) for model_id, score in scored]
    return results


def generate_comparison(dataset="datasetguhring", k=10):
    from src.data.dataset_registry import dataset_artifacts

    artifacts = dataset_artifacts(dataset)
    meta_dir = os.path.join(artifacts.root, "_meta")
    similia_rows = load_similia_rows(meta_dir)
    search_results = compute_cadgcl_search_results(
        dataset,
        [row["query_uuid"] for row in similia_rows],
        k,
    )
    rows = annotate_with_cadgcl(similia_rows, search_results)
    csv_path = os.path.join(meta_dir, "similia_cadgcl_comparison.csv")
    xlsx_path = os.path.join(meta_dir, "similia_cadgcl_comparison.xlsx")
    write_csv_rows(csv_path, OUTPUT_HEADERS, rows)
    write_xlsx_rows(xlsx_path, OUTPUT_HEADERS, rows)
    return csv_path, xlsx_path, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasetguhring")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    csv_path, xlsx_path, rows = generate_comparison(args.dataset, args.k)
    matches = sum(1 for row in rows if row["appears_in_cadgcl_top_k"])
    queries = len({row["query_uuid"] for row in rows})
    print(f"Rows: {len(rows)}")
    print(f"Queries: {queries}")
    print(f"CADGCL matches in top-{args.k}: {matches}")
    print(f"CSV: {csv_path}")
    print(f"XLSX: {xlsx_path}")


if __name__ == "__main__":
    main()
