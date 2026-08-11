import individual_api.runtime_memory_guard as guard
from individual_api import production_runtime_fix


def test_pr31_model_bundle_is_loaded_once(monkeypatch) -> None:
    calls = []
    bundle = {"bundle_version": "pr31-frozen-1"}

    def fake_loader():
        calls.append(1)
        return bundle

    guard._cached_model_bundle.cache_clear()
    monkeypatch.setattr(guard, "_BASE_MODEL_LOADER", fake_loader)
    first = guard._cached_model_bundle()
    second = guard._cached_model_bundle()
    assert first is bundle
    assert second is bundle
    assert len(calls) == 1
    guard._cached_model_bundle.cache_clear()


def test_pdf_text_cache_is_cleared() -> None:
    production_runtime_fix._cached_extract_text.cache_clear()
    assert production_runtime_fix._cached_extract_text.cache_info().currsize == 0
    guard.clear_pdf_text_cache()
    assert production_runtime_fix._cached_extract_text.cache_info().currsize == 0
