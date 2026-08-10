from individual_api.error_resilience import _group_variants


def test_group_variants_recovers_compressed_browser_spacing() -> None:
    # Normal parser requires about 1.28x separation.  This synthetic browser
    # layout has only 1.15x line gaps and must still recover the full lineup.
    found = [1, 5, 2, 3, 9, 7, 6, 8, 4]
    xs = [0.0, 10.0, 21.5, 33.0, 43.0, 53.0, 64.5, 76.0, 86.0]
    variants = _group_variants(found, xs)
    groups = [item[0] for item in variants]
    assert [[1, 5], [2], [3, 9, 7], [6], [8, 4]] in groups


def test_group_variants_rejects_uniform_spacing() -> None:
    found = [1, 2, 3, 4, 5]
    xs = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert _group_variants(found, xs) == []
