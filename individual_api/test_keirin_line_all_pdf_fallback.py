from pathlib import Path

from individual_api import keirin_line_runtime_fix as fix


def test_line_parser_retries_other_selected_pdf(monkeypatch, tmp_path: Path):
    basic = tmp_path / "basic.pdf"
    hs = tmp_path / "hs.pdf"
    odds = tmp_path / "odds.pdf"
    for path in (basic, hs, odds):
        path.write_bytes(b"%PDF-test")

    expected = [[1, 2], [3], [4, 5, 6, 7]]
    monkeypatch.setattr(
        fix,
        "_coordinate_parse",
        lambda path, bikes: expected if path.name == "hs.pdf" else None,
    )
    monkeypatch.setattr(fix, "_text_parse", lambda path, bikes: None)

    assert fix.parse_lines_resilient(basic, list(range(1, 8))) == expected


def test_line_parser_checks_all_sibling_pdfs(monkeypatch, tmp_path: Path):
    paths = [tmp_path / name for name in ("01.pdf", "02.pdf", "03.pdf")]
    for path in paths:
        path.write_bytes(b"%PDF-test")
    calls = []
    monkeypatch.setattr(fix, "_coordinate_parse", lambda path, bikes: calls.append(path.name) or None)
    monkeypatch.setattr(fix, "_text_parse", lambda path, bikes: None)

    try:
        fix.parse_lines_resilient(paths[0], list(range(1, 8)))
    except Exception:
        pass
    assert calls == ["01.pdf", "02.pdf", "03.pdf"]
