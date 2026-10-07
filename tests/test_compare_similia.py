def test_load_similia_rows_combines_xlsx_files(tmp_path):
    from scripts.compare_similia import load_similia_rows, write_xlsx_rows

    headers = ["Name", "Ähnlichkeit"]
    rows = [
        {"Name": "query-a", "Ähnlichkeit": 1.0},
        {"Name": "part-b", "Ähnlichkeit": 0.86},
    ]
    write_xlsx_rows(str(tmp_path / "001.xlsx"), headers, rows)

    similia_rows = load_similia_rows(str(tmp_path))

    assert similia_rows == [
        {
            "query_file": "001.xlsx",
            "query_uuid": "query-a",
            "similia_rank": 1,
            "uuid": "query-a",
            "similia_similarity": 1.0,
        },
        {
            "query_file": "001.xlsx",
            "query_uuid": "query-a",
            "similia_rank": 2,
            "uuid": "part-b",
            "similia_similarity": 0.86,
        },
    ]


def test_annotate_with_cadgcl_adds_rank_score_and_match_flag():
    from scripts.compare_similia import annotate_with_cadgcl

    rows = [
        {"query_uuid": "query-a", "uuid": "query-a"},
        {"query_uuid": "query-a", "uuid": "part-b"},
        {"query_uuid": "query-a", "uuid": "part-c"},
    ]
    search_results = {
        "query-a": [("query-a", 1.0), ("part-b", 0.8)],
    }

    annotated = annotate_with_cadgcl(rows, search_results)

    assert annotated[0]["cadgcl_rank"] == 1
    assert annotated[0]["cadgcl_score"] == 1.0
    assert annotated[0]["appears_in_cadgcl_top_k"] is True
    assert annotated[1]["cadgcl_rank"] == 2
    assert annotated[1]["cadgcl_score"] == 0.8
    assert annotated[1]["appears_in_cadgcl_top_k"] is True
    assert annotated[2]["cadgcl_rank"] == ""
    assert annotated[2]["cadgcl_score"] == ""
    assert annotated[2]["appears_in_cadgcl_top_k"] is False
