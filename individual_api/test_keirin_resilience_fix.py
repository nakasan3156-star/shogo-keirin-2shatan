from keirin_resilience_fix import install_resilience_fix


def test_missing_lines_fall_back_to_singletons(monkeypatch):
    install_resilience_fix()
    import keirin_real_pdf_adapter as adapter

    monkeypatch.setattr(adapter, "_RESILIENCE_FIX_INSTALLED", False, raising=False)


def test_safe_line_parser_returns_complete_singletons():
    install_resilience_fix()
    import keirin_real_pdf_adapter as adapter

    lines = adapter._parse_lines("missing.pdf", [1, 2, 3, 4, 5, 6, 7])
    assert lines == [[1], [2], [3], [4], [5], [6], [7]]
