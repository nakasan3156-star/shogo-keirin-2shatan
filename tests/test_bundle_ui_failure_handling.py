from app.bundle_ui import INDEX_HTML


def test_ui_retries_temporary_gateway_failures_and_has_japanese_fallbacks() -> None:
    assert "[502,503,504].includes(response.status)" in INDEX_HTML
    assert "自動でもう一度試しています" in INDEX_HTML
    assert "AbortController" in INDEX_HTML
    assert "JSON.parse(raw)" in INDEX_HTML
    assert "Unexpected token" not in INDEX_HTML
    assert "サーバー応答を確認できません" in INDEX_HTML
