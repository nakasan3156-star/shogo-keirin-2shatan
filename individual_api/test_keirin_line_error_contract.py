from pathlib import Path

import pytest

from individual_api.keirin_line_runtime_fix import parse_lines_resilient
from individual_api.keirin_pdf_adapter import PdfInputError


def test_failure_reports_all_attempted_pdf_names(monkeypatch, tmp_path: Path):
    from individual_api import keirin_line_runtime_fix as fix
    paths = [tmp_path / name for name in ("a.pdf", "b.pdf", "c.pdf")]
    for path in paths:
        path.write_bytes(b"%PDF-test")
    monkeypatch.setattr(fix, "_coordinate_parse", lambda path, bikes: None)
    monkeypatch.setattr(fix, "_text_parse", lambda path, bikes: None)
    with pytest.raises(PdfInputError) as caught:
        parse_lines_resilient(paths[0], list(range(1, 8)))
    assert caught.value.code == "LINE_PARSE_FAILED_ALL_SOURCES"
    assert set(caught.value.missing) == {"a.pdf", "b.pdf", "c.pdf"}
