"""
schema_generator.py — TrifectAI Schema Induction Engine
════════════════════════════════════════════════════════
Two-stage AI pipeline that reads business-document PDFs and generates a
comprehensive key-value extraction schema for OCI Document Understanding
model training — in the same format as oracle_fusion_receivables_kv_schema.json.

  Stage 1 — Google Gemini Pro  : reads the PDF (multimodal) and induces
                                  an initial schema draft
  Stage 2 — xAI Grok           : reinforces and validates the draft for
                                  completeness, precision, and consistency

Usage:
    python schema_generator.py --pdf invoice.pdf [--pdf invoice2.pdf ...] \
                               [--output schema.json] [--skip-grok] [--draft]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
from pathlib import Path

import oci
import oci.ai_vision
import oci.ai_vision.models
import oci.generative_ai_inference
import oci.generative_ai_inference.models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
log = logging.getLogger("schema_gen")

# ── PDF chunking for OCI Vision's 5-page inline limit ───────────────
try:
    from pypdf import PdfReader, PdfWriter
    _PYPDF_OK = True
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter
        _PYPDF_OK = True
    except ImportError:
        _PYPDF_OK = False

OCI_PAGE_LIMIT = 5

# ═══════════════════════════════════════════════════════════════════
# CONFIG — mirrors trifectai.py; override via environment variables
# ═══════════════════════════════════════════════════════════════════

CFG = {
    "user":        os.getenv("OCI_USER",       "ocid1.user.oc1..aaaaaaaazshw52cia2z2t5642ozk734o5f3vd6myl3y5fabbih22uojz4ilq"),
    "fingerprint": os.getenv("OCI_FINGERPRINT", "a9:5a:d1:80:5d:89:a2:da:0e:fc:27:77:a7:28:a6:d9"),
    "tenancy":     os.getenv("OCI_TENANCY",     "ocid1.tenancy.oc1..aaaaaaaanzcjfb3euqcertohpzzaexzuh5ekzbz6iuhoshufnl6wbnv34ivq"),
    "region":      os.getenv("OCI_REGION",      "us-ashburn-1"),
    "key_file":    os.getenv("OCI_KEY_FILE",    str(Path(__file__).parent / "oci_key.pem")),
}

COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT", CFG["tenancy"])
GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT",
    "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "google.gemini-2.5-pro")
GROK_MODEL   = os.getenv("GROK_MODEL",
    "ocid1.generativeaimodel.oc1.iad.amaaaaaask7dceyaini2prwyi73oia5wdgb2mxgnnrihpocw7hjjgdfkowaa")

VALID_TYPES = {"string", "date", "currency", "number"}

# ═══════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════

_GEMINI_SYSTEM = """\
You are an expert document schema analyst specialising in enterprise business documents
used to train OCI Document Understanding AI models.

Analyse the provided document and generate a comprehensive key-value extraction schema.
For every extractable field produce ONE schema entry:

  "key"         – camelCase identifier  (e.g. InvoiceNumber, BillToAddress, LineItemAmount)
  "dataType"    – exactly one of: "string", "date", "currency", "number"
  "description" – precise extraction instructions that include ALL of:
                    • The exact printed label or column header closest to the field
                    • Spatial location on the page (e.g. "upper-right panel", "footer",
                      "third column of the Line Items table")
                    • Value format with a concrete example
                    • Disambiguation: what NOT to extract ("Do not extract X or Y")
                    • For table columns that repeat per row: state "This field repeats once
                      per [table name] row — return as a list preserving table order, with
                      the same length and ordering as [anchor key]"

Cover all document zones:
  1. Header / metadata  (IDs, dates, reference numbers, status)
  2. Party blocks       (bill-to, ship-to, remit-to: name, address, tax IDs)
  3. Banking / payment  (bank name, account, routing, SWIFT, wire ref)
  4. Line-item table    (one schema entry per column, all marked as repeating)
  5. Tax-detail table   (one schema entry per column, all marked as repeating)
  6. Totals / summary   (subtotal, discounts, taxable, tax, grand total)
  7. Terms / legal      (payment terms, late fees, dispute contact, authorized-by)
  8. Footer             (printed date, transaction source, system metadata)

Return ONLY a valid JSON array — no markdown fences, no explanation:
[{"key":"...","dataType":"...","description":"..."}, ...]"""

_GEMINI_USER_TMPL = """\
## Vision OCR Text (all pages)
```
{ocr_text}
```

{pdf_note}

Generate the complete extraction schema JSON array now."""

_GROK_SYSTEM = """\
You are a senior data-labelling specialist for enterprise document-AI training programmes.
You are given a draft key-value schema produced by another AI and the original document text.
Your role is to REINFORCE and FINALISE the schema so it is production-ready for training
OCI Document Understanding models.

Apply these rules without exception:

1. VALIDATE every entry
   – Is the key unambiguous and camelCase?
   – Is the dataType correct (date for dates, currency for money, number for counts)?
   – Does the description pinpoint the field location with enough precision?

2. ADD every missing field
   – Scan the document text line by line; every labelled value must have a schema entry.

3. REFINE descriptions
   – Add the exact printed label ("Labeled 'Invoice Number' in the …")
   – Add a concrete format example ("Format: DD-Mon-YYYY, e.g. 05-May-2026")
   – Add a "Do not extract …" note for any field easily confused with another.

4. ENFORCE repeating-field rules for ALL table columns
   – Must say "This field repeats once per row — extract … from the [N]th column ('[ColName}')
     of EVERY row in the [Table Name] table."
   – Must say "Return as a list preserving table order, with the same length and
     ordering as [anchor key, e.g. LineNumber]."

5. VERIFY totals arithmetic relationships are noted in the relevant description
   (e.g. "SubTotal + TotalTaxAmount − DiscountTotal = InvoiceTotal").

6. PRESERVE the order: header → parties → banking → line items → tax lines → totals → terms → footer.

Return ONLY a valid JSON array — no markdown, no commentary:
[{"key":"...","dataType":"...","description":"..."}, ...]"""

_GROK_USER_TMPL = """\
## Original Document Text
```
{ocr_text}
```

## Draft Schema from Gemini ({draft_count} fields)
```json
{draft_schema}
```

Produce the final, validated, comprehensive schema array now."""

# ═══════════════════════════════════════════════════════════════════
# PDF CHUNKING
# ═══════════════════════════════════════════════════════════════════

def _split_chunks(pdf_bytes: bytes) -> list[bytes]:
    if not _PYPDF_OK:
        return [pdf_bytes]
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)
    if total <= OCI_PAGE_LIMIT:
        return [pdf_bytes]
    chunks = []
    for start in range(0, total, OCI_PAGE_LIMIT):
        w = PdfWriter()
        for i in range(start, min(start + OCI_PAGE_LIMIT, total)):
            w.add_page(reader.pages[i])
        buf = io.BytesIO()
        w.write(buf)
        chunks.append(buf.getvalue())
    return chunks

# ═══════════════════════════════════════════════════════════════════
# OCI CLIENTS
# ═══════════════════════════════════════════════════════════════════

def _make_clients() -> dict:
    return {
        "genai":  oci.generative_ai_inference.GenerativeAiInferenceClient(
                      config=CFG, service_endpoint=GENAI_ENDPOINT),
        "vision": oci.ai_vision.AIServiceVisionClient(config=CFG),
    }

# ═══════════════════════════════════════════════════════════════════
# VISION OCR  —  text extraction from PDF
# ═══════════════════════════════════════════════════════════════════

def _ocr_chunk(clients: dict, chunk: bytes) -> str:
    b64 = base64.b64encode(chunk).decode()
    resp = clients["vision"].analyze_document(
        analyze_document_details=oci.ai_vision.models.AnalyzeDocumentDetails(
            document=oci.ai_vision.models.InlineDocumentDetails(source="INLINE", data=b64),
            features=[oci.ai_vision.models.DocumentTextDetectionFeature()],
            compartment_id=COMPARTMENT_ID,
            language="ENG",
        )
    )
    lines = []
    for pg in getattr(resp.data, "pages", []) or []:
        for ln in getattr(pg, "lines", []) or []:
            t = getattr(ln, "text", "") or ""
            if t.strip():
                lines.append(t.strip())
    return "\n".join(lines)


def extract_ocr(clients: dict, pdf_bytes: bytes) -> str:
    """OCR all pages, return full text."""
    chunks = _split_chunks(pdf_bytes)
    log.info("[OCR] %d page-chunk(s) to process", len(chunks))
    parts = []
    for i, chunk in enumerate(chunks, 1):
        log.info("[OCR] Chunk %d/%d", i, len(chunks))
        parts.append(_ocr_chunk(clients, chunk))
    text = "\n".join(p for p in parts if p)
    log.info("[OCR] %d chars extracted", len(text))
    return text

# ═══════════════════════════════════════════════════════════════════
# GENAI HELPERS
# ═══════════════════════════════════════════════════════════════════

def _genai_chat(clients: dict, model_id: str,
                system_text: str, user_contents: list) -> str:
    """Send a chat request to any OCI GenAI model and return the response text."""
    Msg  = oci.generative_ai_inference.models
    req  = Msg.GenericChatRequest(
        api_format=Msg.BaseChatRequest.API_FORMAT_GENERIC,
        messages=[
            Msg.SystemMessage(
                role="SYSTEM",
                content=[Msg.TextContent(type="TEXT", text=system_text)],
            ),
            Msg.UserMessage(
                role="USER",
                content=user_contents,
            ),
        ],
        max_tokens=8192,
        temperature=0.0,
        is_stream=False,
    )
    resp = clients["genai"].chat(
        Msg.ChatDetails(
            compartment_id=COMPARTMENT_ID,
            serving_mode=Msg.OnDemandServingMode(model_id=model_id),
            chat_request=req,
        )
    )
    return resp.data.chat_response.choices[0].message.content[0].text


def _parse_schema(raw: str, source: str) -> list[dict]:
    """Extract, parse, and validate a JSON array of schema entries from an LLM response."""
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.M)
    clean = re.sub(r"\s*```\s*$", "", clean.strip(), flags=re.M)

    parsed = None
    for attempt in (clean, re.search(r"\[.*\]", clean, re.DOTALL)):
        candidate = attempt if isinstance(attempt, str) else (attempt.group() if attempt else None)
        if candidate:
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

    if not isinstance(parsed, list):
        log.error("[%s] Could not parse JSON array from response", source)
        return []

    out = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        key  = str(entry.get("key", "")).strip()
        dt   = str(entry.get("dataType", "string")).strip()
        desc = str(entry.get("description", "")).strip()
        if not key or not desc:
            continue
        if dt not in VALID_TYPES:
            dt = "string"
        out.append({"key": key, "dataType": dt, "description": desc})

    log.info("[%s] Parsed %d valid schema fields", source, len(out))
    return out

# ═══════════════════════════════════════════════════════════════════
# STAGE 1 — Gemini: read PDF + induce initial schema
# ═══════════════════════════════════════════════════════════════════

def run_gemini(clients: dict, ocr_text: str, pdf_bytes: bytes) -> list[dict]:
    log.info("[Gemini] Stage 1 — inducing schema draft...")
    Msg = oci.generative_ai_inference.models

    # Attempt to send the PDF inline for Gemini's native multimodal PDF understanding.
    # OCI GenAI proxies the data URL directly to Gemini, which accepts application/pdf.
    user_contents = []
    pdf_note = ""
    try:
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        user_contents.append(
            Msg.ImageContent(
                type="IMAGE",
                image_url=Msg.ImageUrl(
                    url=f"data:application/pdf;base64,{b64_pdf}"
                ),
            )
        )
        pdf_note = "The PDF is attached inline above — use its visual layout to identify precise field locations, labels, and table structure."
        log.info("[Gemini] PDF attached inline (%d KB)", len(pdf_bytes) // 1024)
    except Exception as exc:
        pdf_note = "No inline PDF available — infer layout from OCR text patterns (colons, label-value pairs, table column headers)."
        log.warning("[Gemini] Inline PDF not supported (%s) — text-only mode", exc)

    user_contents.append(
        Msg.TextContent(
            type="TEXT",
            text=_GEMINI_USER_TMPL.format(
                ocr_text=ocr_text[:8000],
                pdf_note=pdf_note,
            ),
        )
    )

    raw = _genai_chat(clients, GEMINI_MODEL, _GEMINI_SYSTEM, user_contents)
    return _parse_schema(raw, "Gemini")

# ═══════════════════════════════════════════════════════════════════
# STAGE 2 — Grok: reinforce and validate
# ═══════════════════════════════════════════════════════════════════

def run_grok(clients: dict, ocr_text: str, draft: list[dict]) -> list[dict]:
    log.info("[Grok] Stage 2 — reinforcing schema (%d draft fields)...", len(draft))
    Msg = oci.generative_ai_inference.models

    user_text = _GROK_USER_TMPL.format(
        ocr_text=ocr_text[:8000],
        draft_count=len(draft),
        draft_schema=json.dumps(draft, indent=2),
    )
    raw = _genai_chat(
        clients, GROK_MODEL, _GROK_SYSTEM,
        [Msg.TextContent(type="TEXT", text=user_text)],
    )
    return _parse_schema(raw, "Grok")

# ═══════════════════════════════════════════════════════════════════
# MERGE — Grok is authoritative; Gemini fills any gaps Grok dropped
# ═══════════════════════════════════════════════════════════════════

def merge_schemas(gemini: list[dict], grok: list[dict]) -> list[dict]:
    grok_keys = {e["key"] for e in grok}
    merged = list(grok)
    added = 0
    for entry in gemini:
        if entry["key"] not in grok_keys:
            merged.append(entry)
            added += 1
            log.info("[Merge] Gemini-only field carried forward: %s", entry["key"])
    log.info("[Merge] Final: %d fields (%d from Grok + %d Gemini-only)",
             len(merged), len(grok), added)
    return merged

# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def generate_schema(pdf_paths: list[Path], output_path: Path,
                    skip_grok: bool = False, draft_only: bool = False) -> None:
    clients = _make_clients()

    # ── OCR all PDFs ────────────────────────────────────────────────
    ocr_parts: list[str] = []
    primary_pdf: bytes = b""

    for pdf_path in pdf_paths:
        log.info("[Input] %s", pdf_path.name)
        pdf_bytes = pdf_path.read_bytes()
        if not primary_pdf:
            primary_pdf = pdf_bytes        # first PDF goes to Gemini multimodal
        ocr_text = extract_ocr(clients, pdf_bytes)
        ocr_parts.append(f"=== {pdf_path.name} ===\n{ocr_text}")

    combined_ocr = "\n\n".join(ocr_parts)

    # ── Stage 1: Gemini ─────────────────────────────────────────────
    gemini_schema = run_gemini(clients, combined_ocr, primary_pdf)
    if not gemini_schema:
        log.error("Gemini returned no schema fields — cannot continue")
        sys.exit(1)

    if draft_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(gemini_schema, indent=2), encoding="utf-8")
        print(f"\nDraft schema (Gemini only): {output_path} — {len(gemini_schema)} fields")
        return

    if skip_grok:
        final_schema = gemini_schema
        log.info("[Pipeline] Grok stage skipped — using Gemini draft as final output")
    else:
        # ── Stage 2: Grok ────────────────────────────────────────────
        grok_schema = run_grok(clients, combined_ocr, gemini_schema)
        if not grok_schema:
            log.warning("[Pipeline] Grok returned no fields — falling back to Gemini draft")
            final_schema = gemini_schema
        else:
            final_schema = merge_schemas(gemini_schema, grok_schema)

    # ── Write output ─────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_schema, indent=2), encoding="utf-8")
    log.info("[Done] %s written — %d fields", output_path, len(final_schema))
    print(f"\nSchema written: {output_path} — {len(final_schema)} fields")

# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TrifectAI Schema Generator\n"
            "Reads PDFs with Gemini Pro, then reinforces the schema with Grok 4.3\n"
            "to produce a key-value labeling schema for OCI Document Understanding."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pdf", dest="pdfs", action="append", required=True, metavar="PATH",
        help="Path to a PDF document (repeat flag for multiple documents)",
    )
    p.add_argument(
        "--output", default="generated_schema.json", metavar="PATH",
        help="Output JSON schema file (default: generated_schema.json)",
    )
    p.add_argument(
        "--skip-grok", action="store_true",
        help="Skip the Grok reinforcement stage and output the Gemini draft directly",
    )
    p.add_argument(
        "--draft", action="store_true",
        help="Alias for --skip-grok; also writes to output path with '(draft)' label",
    )
    p.add_argument(
        "--gemini-model", default=None, metavar="MODEL_ID",
        help=f"Override Gemini model ID (default: {GEMINI_MODEL})",
    )
    p.add_argument(
        "--grok-model", default=None, metavar="MODEL_OCID",
        help="Override Grok model OCID",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    global GEMINI_MODEL, GROK_MODEL
    if args.gemini_model:
        GEMINI_MODEL = args.gemini_model
    if args.grok_model:
        GROK_MODEL = args.grok_model

    pdf_paths: list[Path] = []
    for raw in args.pdfs:
        p = Path(raw)
        if not p.exists():
            print(f"Error: file not found — {p}", file=sys.stderr)
            sys.exit(1)
        if p.suffix.lower() != ".pdf":
            print(f"Error: not a PDF file — {p}", file=sys.stderr)
            sys.exit(1)
        pdf_paths.append(p)

    generate_schema(
        pdf_paths=pdf_paths,
        output_path=Path(args.output),
        skip_grok=args.skip_grok or args.draft,
        draft_only=args.draft,
    )


if __name__ == "__main__":
    main()
