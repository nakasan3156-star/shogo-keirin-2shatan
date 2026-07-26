"""実PDFアダプターのテスト。

使用法:
python test_keirin_pdf_adapter.py 出走表.pdf HS.pdf オッズ.pdf
"""

from __future__ import annotations

import json
import sys

from keirin_pdf_adapter import predict_from_files


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("出走表PDF、H・S PDF、2車単オッズPDFを指定してください")
    first = predict_from_files(*sys.argv[1:4])
    second = predict_from_files(*sys.argv[1:4])
    assert first == second
    assert first["status"] == "OK", json.dumps(first, ensure_ascii=False)
    assert first["simulations"] == 100_000
    assert first["pdf_audit"]["result_data_used"] is False
    assert first["pdf_audit"]["web_data_used"] is False
    rider_count = first["pdf_audit"]["rider_count"]
    assert first["pdf_audit"]["odds_count"] == rider_count * (rider_count - 1)
    print(json.dumps({
        "status": first["status"],
        "version": first["version"],
        "race": first["pdf_audit"]["race"],
        "lines": first["pdf_audit"]["lines"],
        "simulations": first["simulations"],
        "candidates": first["candidates"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
