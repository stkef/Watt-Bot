from wattbot.chunk import chunk_page, chunk_document, MAX_TABLE_WORDS


def test_windows_and_overlap_stay_on_page():
    text = " ".join(f"w{i}" for i in range(600))
    ch = chunk_page("doc", 3, text)
    assert all(c.page == 3 and c.doc_id == "doc" for c in ch)
    assert len(ch[0].text.split()) == 260
    assert ch[1].text.split()[0] == "w210"  # 50-word overlap


def test_table_kept_whole_with_header():
    table = "| model | MWh |\n| --- | --- |\n| GPT-3 | 1287 |\n| BLOOM | 433 |"
    ch = chunk_page("doc", 1, "Intro text here.\n" + table + "\nAfter text.")
    tables = [c for c in ch if c.kind == "table"]
    assert len(tables) == 1 and "| model | MWh |" in tables[0].text and "BLOOM" in tables[0].text


def test_big_table_split_repeats_header():
    rows = "\n".join(f"| m{i} | " + "x " * 20 + "|" for i in range(100))
    table = "| model | value |\n| --- | --- |\n" + rows
    parts = [c for c in chunk_page("d", 1, table) if c.kind == "table"]
    assert len(parts) > 1
    assert all(p.text.startswith("| model | value |\n| --- | --- |") for p in parts)
    assert all(len(p.text.split()) <= MAX_TABLE_WORDS + 40 for p in parts)


def test_no_chunk_crosses_pages():
    ch = chunk_document("d", [{"page": 1, "text": "alpha " * 300}, {"page": 2, "text": "beta " * 300}])
    for c in ch:
        assert ("alpha" in c.text) != ("beta" in c.text)
