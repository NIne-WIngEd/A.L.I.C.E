from __future__ import annotations

import argparse
import html
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.judge_calibration import (
    load_calibration_bundle,
    load_judge_calibration_policy,
    save_human_label,
)


def latest_bundle(vault: Path, pilot_name: str) -> Path:
    directory = vault / "manifests" / "calibration" / pilot_name
    candidates = sorted(
        directory.glob("judge-calibration-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No judge-calibration bundle was found")
    return candidates[0]


def esc(value) -> str:
    return html.escape(str(value or ""))


def render_page(bundle_path: Path, bundle: dict) -> str:
    items = list(bundle.get("items", []))
    labeled = sum(
        bool(str(item.get("human_label", "")))
        for item in items
    )
    current_index = next(
        (
            index
            for index, item in enumerate(items)
            if not str(item.get("human_label", ""))
        ),
        None,
    )

    if current_index is None:
        body = f"""
        <main>
          <h1>Calibration complete</h1>
          <p>You labeled {labeled} of {len(items)} items.</p>
          <p>Return to PowerShell, stop the server with Ctrl+C, and run the evaluation command.</p>
          <p class="path">{esc(bundle_path)}</p>
        </main>
        """
    else:
        item = items[current_index]
        evidence_html = []

        for window in item.get("evidence_windows", []):
            source_line = " · ".join(
                value
                for value in [
                    str(window.get("citation", "")),
                    str(window.get("filename", "")),
                    str(window.get("family", "")),
                    str(window.get("owner_relation", "")),
                ]
                if value
            )
            evidence_html.append(
                f"""
                <section class="evidence">
                  <div class="source">{esc(source_line)}</div>
                  <pre>{esc(window.get("text", ""))}</pre>
                </section>
                """
            )

        body = f"""
        <main>
          <div class="progress">Item {current_index + 1} of {len(items)} · {labeled} labeled</div>

          <h2>Question</h2>
          <div class="card question">{esc(item.get("question", ""))}</div>

          <h2>Claim</h2>
          <div class="card claim">{esc(item.get("claim_text", ""))}</div>

          <h2>Cited evidence</h2>
          {''.join(evidence_html)}

          <form method="post" action="/label">
            <input type="hidden" name="item_id" value="{esc(item.get("item_id", ""))}">
            <div class="buttons">
              <button class="supported" name="label" value="supported">Supported</button>
              <button class="partial" name="label" value="partially_supported">Partially supported</button>
              <button class="unsupported" name="label" value="unsupported">Unsupported</button>
            </div>
          </form>

          <details>
            <summary>Review criteria</summary>
            <p><strong>Supported:</strong> every material assertion in the claim is established by the cited evidence.</p>
            <p><strong>Partially supported:</strong> at least one material assertion is supported, but another material assertion is not established.</p>
            <p><strong>Unsupported:</strong> a critical assertion is not established, is only topically related, or conflicts with the evidence.</p>
          </details>
        </main>
        """

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>A.L.I.C.E. Judge Calibration</title>
<style>
body {{
  margin: 0;
  font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  background: #111;
  color: #eee;
}}
main {{
  max-width: 980px;
  margin: 0 auto;
  padding: 36px 24px 80px;
}}
h1, h2 {{ font-weight: 650; }}
.progress {{ color: #aaa; margin-bottom: 28px; }}
.card, .evidence {{
  border: 1px solid #333;
  border-radius: 10px;
  padding: 18px;
  margin-bottom: 18px;
  background: #181818;
}}
.claim {{ font-size: 1.15rem; }}
.source {{ color: #aaa; margin-bottom: 10px; font-size: .9rem; }}
pre {{
  white-space: pre-wrap;
  font-family: inherit;
  margin: 0;
  line-height: 1.55;
}}
.buttons {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin: 28px 0;
}}
button {{
  border: 0;
  border-radius: 8px;
  padding: 14px 20px;
  font-size: 1rem;
  cursor: pointer;
}}
.supported {{ background: #238636; color: white; }}
.partial {{ background: #9e6a03; color: white; }}
.unsupported {{ background: #da3633; color: white; }}
details {{ color: #bbb; margin-top: 26px; }}
.path {{ word-break: break-all; color: #aaa; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    bundle_path: Path

    def _send(self, status: int, content: str, content_type: str = "text/html; charset=utf-8") -> None:
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path not in {"/", "/index.html"}:
            self._send(404, "Not found", "text/plain; charset=utf-8")
            return
        bundle = load_calibration_bundle(self.bundle_path)
        self._send(200, render_page(self.bundle_path, bundle))

    def do_POST(self):
        if self.path != "/label":
            self._send(404, "Not found", "text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        values = urllib.parse.parse_qs(raw)

        item_id = values.get("item_id", [""])[0]
        label = values.get("label", [""])[0]

        try:
            save_human_label(
                bundle_path=self.bundle_path,
                item_id=item_id,
                label=label,
            )
        except Exception as exc:
            self._send(
                400,
                esc(f"{type(exc).__name__}: {exc}"),
                "text/plain; charset=utf-8",
            )
            return

        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, format, *args):
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    vault = args.vault.expanduser().resolve(strict=True)
    policy = load_judge_calibration_policy()

    bundle_path = (
        args.calibration.expanduser().resolve(strict=True)
        if args.calibration
        else latest_bundle(vault, args.pilot_name)
    )

    host = args.host or policy.review_host
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Review server may only bind to loopback")

    port = args.port or policy.review_port
    Handler.bundle_path = bundle_path

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"

    print("A.L.I.C.E. judge-calibration review")
    print(f"Bundle: {bundle_path}")
    print(f"Open: {url}")
    print("Press Ctrl+C when review is complete.")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped.")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
