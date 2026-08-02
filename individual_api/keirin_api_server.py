"""標準ライブラリだけで動く2方式競輪APIサーバー。"""

from __future__ import annotations

import json
import os
import tempfile
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from keirin_dual_strategy_api import VERSION, predict
from keirin_dual_pdf_adapter import predict_from_files
from keirin_pdf_adapter import REQUIRED_UPLOADS


MAX_BODY_BYTES = 50_000_000


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {
                "status": "ok",
                "version": VERSION,
                "odds_source": "KEIRIN.JP",
                "strategies": {"shogo": 5, "residual": 3},
            })
        else:
            self._send(404, {"status": "NOT_FOUND"})

    def do_POST(self) -> None:
        if self.path not in {"/predict", "/predict-files"}:
            self._send(404, {"status": "NOT_FOUND"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit = MAX_BODY_BYTES if self.path == "/predict-files" else 5_000_000
            if length <= 0 or length > limit:
                self._send(400, {
                    "status": "INPUT_ERROR",
                    "purchase_status": "NO_BET",
                    "error": {"code": "INVALID_BODY_SIZE", "message": "本文サイズが不正です", "missing": []},
                })
                return
            body = self.rfile.read(length)
            if self.path == "/predict":
                result = predict(json.loads(body))
            else:
                result = self._predict_files(body)
            self._send(200 if result["status"] == "OK" else 422, result)
        except (json.JSONDecodeError, TypeError, ValueError):
            self._send(400, {
                "status": "INPUT_ERROR",
                "purchase_status": "NO_BET",
                "error": {"code": "INVALID_JSON", "message": "JSONを読み取れません", "missing": []},
            })
        except Exception:
            self._send(500, {
                "status": "PROCESSING_ERROR",
                "purchase_status": "NO_BET",
                "error": {
                    "code": "UNEXPECTED_SERVER_ERROR",
                    "message": "APIを安全停止しました",
                    "missing": [],
                },
            })

    def _predict_files(self, body: bytes) -> dict:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            return {
                "version": VERSION,
                "status": "INPUT_ERROR",
                "purchase_status": "NO_BET",
                "error": {
                    "code": "MULTIPART_REQUIRED",
                    "message": "multipart/form-dataで3PDFを送信してください",
                    "missing": list(REQUIRED_UPLOADS),
                },
            }
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("ascii", "ignore")
            + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
        )
        uploads: dict[str, tuple[str, bytes]] = {}
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            if name not in {*REQUIRED_UPLOADS, "ex_image"}:
                continue
            data = part.get_payload(decode=True) or b""
            filename = Path(part.get_filename() or name).name
            uploads[name] = (filename, data)
        missing = [name for name in REQUIRED_UPLOADS if name not in uploads]
        if missing:
            return {
                "version": VERSION,
                "status": "INPUT_ERROR",
                "purchase_status": "NO_BET",
                "error": {
                    "code": "MISSING_UPLOAD",
                    "message": "必須PDFが不足しています",
                    "missing": missing,
                },
            }
        with tempfile.TemporaryDirectory(prefix="keirin-api-") as directory:
            saved: dict[str, Path] = {}
            for name, (filename, data) in uploads.items():
                suffix = Path(filename).suffix or (".jpg" if name == "ex_image" else ".pdf")
                destination = Path(directory) / f"{name}{suffix}"
                destination.write_bytes(data)
                saved[name] = destination
            return predict_from_files(
                saved["racecard_pdf"],
                saved["hs_pdf"],
                saved["odds_pdf"],
                saved.get("ex_image"),
            )

    def log_message(self, format: str, *args: object) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Keirin API {VERSION}: http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve(
        host=os.environ.get("KEIRIN_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("KEIRIN_API_PORT", "8787")),
    )
