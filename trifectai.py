"""
trifectai.py  —  TriVōx Invoice Intelligence Engine
═══════════════════════════════════════════════════
Three OCI engines, one consensus verdict.

  • OCI Document Understanding  (Custom KV model)
  • OCI Vision                  (TEXT_DETECTION OCR)
  • OCI Generative AI           (Gemini 2.5 Pro)

Run:
    pip install flask flask-cors oci pdf2image Pillow
    python trifectai.py

Then open: http://localhost:8000
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import logging
import os
import re
import threading
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import oci
import oci.ai_document
import oci.ai_document.models
import oci.ai_vision
import oci.ai_vision.models
import oci.generative_ai_inference
import oci.generative_ai_inference.models
import oci.retry
from authlib.integrations.flask_client import OAuth
from flask import Flask, jsonify, redirect, request, Response, send_from_directory, session, url_for
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
log = logging.getLogger("trifectai")

# ── PDF chunking (OCI inline analyzeDocument hard-limits to 5 pages) ─
try:
    from pypdf import PdfReader, PdfWriter
    _PYPDF_OK = True
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter      # older alias
        _PYPDF_OK = True
    except ImportError:
        _PYPDF_OK = False

OCI_PAGE_LIMIT = 5   # OCI hard limit for inline analyzeDocument

# ── Supported file formats ────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
IMAGE_EXTENSIONS     = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

def image_to_pdf(image_bytes: bytes) -> bytes:
    """
    Convert image bytes (JPG, PNG, TIFF, BMP, WEBP) to a single-page PDF
    using Pillow so that all three OCI engines (DU, Vision, Gemini) can
    process the file identically to a native PDF.

    Falls back gracefully — if Pillow is not installed this raises ImportError
    and the caller will skip DU and send the raw image bytes to Vision instead.
    """
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    # OCI requires RGB; convert palette/RGBA/greyscale images first
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PDF")
    log.info("[ImageConvert] Converted image (%s, %dx%d) → PDF (%d KB)",
             img.format or "?", img.width, img.height, buf.tell() // 1024)
    return buf.getvalue()

def split_pdf_chunks(pdf_bytes: bytes) -> list[bytes]:
    """
    Split a PDF into chunks of at most OCI_PAGE_LIMIT pages each.
    Returns a list of PDF byte strings.  If pypdf is unavailable or the
    document is within the limit, returns [pdf_bytes] unchanged.
    """
    if not _PYPDF_OK:
        log.warning("pypdf not installed — cannot split PDF; large docs may fail")
        return [pdf_bytes]

    reader = PdfReader(io.BytesIO(pdf_bytes))
    total  = len(reader.pages)
    log.info("[PDF] %d total pages — chunk size %d", total, OCI_PAGE_LIMIT)

    if total <= OCI_PAGE_LIMIT:
        return [pdf_bytes]

    chunks = []
    for start in range(0, total, OCI_PAGE_LIMIT):
        writer = PdfWriter()
        for i in range(start, min(start + OCI_PAGE_LIMIT, total)):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())
        log.info("[PDF] Chunk %d: pages %d–%d", len(chunks), start + 1,
                 min(start + OCI_PAGE_LIMIT, total))
    return chunks


def merge_du_results(results: list[dict]) -> dict:
    """
    Merge parse_du() outputs from multiple chunks into one.
    Scalar fields: first non-empty value wins (page 1 is authoritative).
    Array fields (line_items, tax_lines): concatenated in chunk order.
    """
    if not results:
        return _empty_out()
    if len(results) == 1:
        return results[0]

    merged = _empty_out()
    for chunk in results:
        for section in ("header", "parties", "banking", "totals", "terms", "metadata"):
            for k, v in chunk.get(section, {}).items():
                if k not in merged[section]:
                    merged[section][k] = v
        merged["line_items"].extend(chunk.get("line_items", []))
        merged["tax_lines"].extend(chunk.get("tax_lines", []))
    return merged


def merge_vision_results(results: list[dict]) -> dict:
    """
    Merge parse_vision() outputs from multiple chunks.
    Concatenates full_text and increments page_count.
    Scalar hint fields: first non-empty value wins.
    """
    if not results:
        return _empty_out()
    if len(results) == 1:
        return results[0]

    merged = _empty_out()
    merged["full_text"]  = ""
    merged["page_count"] = 0
    texts = []
    for chunk in results:
        texts.append(chunk.get("full_text", ""))
        merged["page_count"] += chunk.get("page_count", 0)
        for section in ("header", "banking", "totals"):
            for k, v in chunk.get(section, {}).items():
                if k not in merged[section]:
                    merged[section][k] = v
    merged["full_text"] = "\n".join(t for t in texts if t)
    return merged

# ═══════════════════════════════════════════════════════════════════
# CONFIG  — edit these or override via environment variables
# ═══════════════════════════════════════════════════════════════════

CFG = {
    "user":               os.getenv("OCI_USER",        "ocid1.user.oc1..aaaaaaaazshw52cia2z2t5642ozk734o5f3vd6myl3y5fabbih22uojz4ilq"),
    "fingerprint":        os.getenv("OCI_FINGERPRINT",  "a9:5a:d1:80:5d:89:a2:da:0e:fc:27:77:a7:28:a6:d9"),
    "tenancy":            os.getenv("OCI_TENANCY",      "ocid1.tenancy.oc1..aaaaaaaanzcjfb3euqcertohpzzaexzuh5ekzbz6iuhoshufnl6wbnv34ivq"),
    "region":             os.getenv("OCI_REGION",       "us-ashburn-1"),
    "key_file":           os.getenv("OCI_KEY_FILE",     "oci_key.pem"),
}

COMPARTMENT_ID      = os.getenv("OCI_COMPARTMENT",   CFG["tenancy"])
NAMESPACE           = os.getenv("OCI_NAMESPACE",     "id0sajugd5y6")
BUCKET              = os.getenv("OCI_BUCKET",        "DocumentAIBucket")

CUSTOM_MODEL_OCID   = os.getenv("DU_MODEL_OCID",
    "ocid1.aidocumentmodel.oc1.us-chicago-1.amaaaaaatfjxduaabalpgtjglbpmjywkkggl7ednnjc3ilfazqfs3na3fcrq")
DU_REGION           = os.getenv("DU_REGION",         "us-chicago-1")

GEMINI_MODEL        = os.getenv("GEMINI_MODEL",      "google.gemini-2.5-pro")
GROK_MODEL          = os.getenv("GROK_MODEL",
    "ocid1.generativeaimodel.oc1.iad.amaaaaaask7dceyaini2prwyi73oia5wdgb2mxgnnrihpocw7hjjgdfkowaa")
GENAI_ENDPOINT      = os.getenv("GENAI_ENDPOINT",
    "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com")

# ── OIDC / Identity Domain ──────────────────────────────────────────
OIDC_DOMAIN_URL    = os.getenv("OIDC_DOMAIN_URL",    "https://idcs-20733d446269488eafe63ae1a6ad6fb3.identity.oraclecloud.com:443")
OIDC_CLIENT_ID     = os.getenv("OIDC_CLIENT_ID",     "8c65f7cf3ad64b34be76c7a4f79011c7")
OIDC_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET", "idcscs-0fc66d14-4a22-4835-a10c-0bd10712dbde")
APP_HOST           = os.getenv("APP_HOST",            "http://127.0.0.1:8000")
SECRET_KEY         = os.getenv("SECRET_KEY",          "change-me-in-production")

# ═══════════════════════════════════════════════════════════════════
# OCI CLIENT FACTORY
# ═══════════════════════════════════════════════════════════════════

def _base_config() -> dict:
    return dict(CFG)

def _du_config() -> dict:
    c = dict(CFG)
    c["region"] = DU_REGION
    return c

_GENAI_TIMEOUT = (10, 300)   # (connect_sec, read_sec) — Grok / Gemini can be slow

def _clients():
    bc = _base_config()
    dc = _du_config()
    genai = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=bc, service_endpoint=GENAI_ENDPOINT)
    # Override the default 60-second read timeout; large LLM responses can take minutes.
    # NoneRetryStrategy disables automatic retries so a timeout fails immediately
    # instead of retrying ~10 times and blocking for 10+ minutes.
    genai.base_client.timeout = _GENAI_TIMEOUT
    genai.retry_strategy = oci.retry.NoneRetryStrategy()
    return {
        # Synchronous DU client for analyze_document (inline bytes, no Object Storage)
        "doc_sync": oci.ai_document.AIServiceDocumentClient(config=dc),
        "vision":   oci.ai_vision.AIServiceVisionClient(config=bc),
        "genai":    genai,
    }

# ═══════════════════════════════════════════════════════════════════
# PARSERS
# ═══════════════════════════════════════════════════════════════════

def _cv(value, conf=1.0):
    return {"value": value, "confidence": conf}

def _safe_arr(v):
    if isinstance(v, list): return v
    try:
        p = json.loads(v)
        if isinstance(p, list): return p
    except Exception: pass
    return [v] if v else []

_DU_MAP = {
    "InvoiceNumber":("header","invoice_number"), "InvoiceDate":("header","invoice_date"),
    "DueDate":("header","due_date"), "PaymentTerms":("terms","payment_terms"),
    "Currency":("header","currency"), "BusinessUnit":("header","business_unit"),
    "TransactionType":("header","transaction_type"), "PurchaseOrderNumber":("header","po_number"),
    "Salesperson":("header","salesperson"), "BillToName":("parties","bill_to_name"),
    "BillToAddress":("parties","bill_to_address"),
    "BillToTaxRegistrationNumber":("parties","bill_to_tax_reg"),
    "ShipToName":("parties","ship_to_name"), "ShipToAddress":("parties","ship_to_address"),
    "RemitToName":("banking","remit_to_name"), "RemitToAddress":("banking","remit_to_address"),
    "BankName":("banking","bank_name"), "BankAccountNumber":("banking","bank_account"),
    "BankRoutingNumber":("banking","routing_number"), "SWIFTCode":("banking","swift_code"),
    "WireReference":("banking","wire_reference"), "SubTotal":("totals","subtotal"),
    "DiscountTotal":("totals","discount_total"), "TaxableAmount":("totals","taxable_amount"),
    "TotalTaxAmount":("totals","total_tax"), "InvoiceTotal":("totals","invoice_total"),
    "LatePaymentFee":("terms","late_fee"), "DisputeContactEmail":("terms","dispute_email"),
    "InvoiceStatus":("terms","invoice_status"), "PrintedDate":("metadata","printed_date"),
    "TransactionSource":("metadata","transaction_source"), "AuthorizedBy":("metadata","authorized_by"),
}
_DU_LINE_KEYS   = ["LineNumber","LineItemDescription","LineItemUOM","LineItemQuantity","LineItemUnitPrice","LineItemDiscountPercent","LineItemAmount"]
_LINE_OUT_KEYS  = ["line_number","description","uom","quantity","unit_price","discount_pct","amount"]
_DU_TAX_KEYS    = ["TaxRegime","TaxName","TaxRateCode","TaxLineTaxableAmount","TaxLineRate","TaxLineAmount"]
_TAX_OUT_KEYS   = ["regime","name","rate_code","taxable_amount","rate_pct","tax_amount"]

def _empty_out():
    return {"header":{},"parties":{},"banking":{},"line_items":[],"tax_lines":[],"totals":{},"terms":{},"metadata":{}}

def parse_du(raw: dict) -> dict:
    out = _empty_out()
    la = {k:[] for k in _DU_LINE_KEYS}; lc = {k:[] for k in _DU_LINE_KEYS}
    ta = {k:[] for k in _DU_TAX_KEYS};  tc = {k:[] for k in _DU_TAX_KEYS}
    for page in raw.get("pages",[]):
        for f in page.get("documentFields",[]):
            name = f.get("fieldLabel",{}).get("name","")
            conf = f.get("fieldLabel",{}).get("confidence",1.0)
            val  = f.get("fieldValue",{}).get("value","")
            if not val: continue
            if name in _DU_LINE_KEYS:
                p = _safe_arr(val); la[name].extend(p); lc[name].extend([conf]*len(p)); continue
            if name in _DU_TAX_KEYS:
                p = _safe_arr(val); ta[name].extend(p); tc[name].extend([conf]*len(p)); continue
            if name in _DU_MAP:
                sec, key = _DU_MAP[name]
                if key not in out[sec]: out[sec][key] = _cv(val, conf)
    for i in range(len(la.get("LineNumber",[]))):
        out["line_items"].append({ok: _cv(la[dk][i] if i<len(la[dk]) else "", lc[dk][i] if i<len(lc[dk]) else 0.5)
                                   for dk,ok in zip(_DU_LINE_KEYS,_LINE_OUT_KEYS)})
    for i in range(len(ta.get("TaxRegime",[]))):
        out["tax_lines"].append({ok: _cv(ta[dk][i] if i<len(ta[dk]) else "", tc[dk][i] if i<len(tc[dk]) else 0.5)
                                  for dk,ok in zip(_DU_TAX_KEYS,_TAX_OUT_KEYS)})
    return out

def parse_vision(data, fallback=False) -> dict:
    out = _empty_out(); out["full_text"]=""; out["page_count"]=0
    lines = []
    if fallback:
        pages = getattr(data,"pages",[]) or []
        out["page_count"] = len(pages)
        for pg in pages:
            for ln in (getattr(pg,"lines",[]) or []):
                t = getattr(ln,"text","") or ""
                if t.strip(): lines.append(t.strip())
    else:
        out["page_count"] = len(data)
        for pn in sorted(data.keys()):
            it = getattr(data[pn],"image_text",None)
            if it:
                for ln in (getattr(it,"lines",[]) or []):
                    t = getattr(ln,"text","") or ""
                    if t.strip(): lines.append(t.strip())
    out["full_text"] = "\n".join(lines)
    txt = out["full_text"]
    def _find(pat,g=1):
        m = re.search(pat,txt,re.I|re.M)
        return m.group(g).strip() if m else ""
    for pat,sec,key,conf in [
        (r"INV[-–]\d{4}[-–]\d{4,6}","header","invoice_number",0.75),
        (r"Invoice Date[:\s]+(\d{2}-\w{3}-\d{4})","header","invoice_date",0.75),
        (r"Due Date[:\s]+(\d{2}-\w{3}-\d{4})","header","due_date",0.75),
        (r"PO[-–]\d{4}[-–]\d{4,6}","header","po_number",0.75),
    ]:
        v = re.search(pat,txt,re.I)
        if v: out[sec][key] = _cv(v.group(0) if "(" not in pat else v.group(1), conf)
    td = _find(r"TOTAL AMOUNT DUE[:\s]+USD\s*([\d,]+\.\d{2})")
    if td: out["totals"]["invoice_total"] = _cv(f"USD {td}", 0.70)
    st = _find(r"Subtotal[:\s]+USD\s*([\d,]+\.\d{2})")
    if st: out["totals"]["subtotal"] = _cv(f"USD {st}", 0.70)
    return out

def parse_gemini(raw_text: str) -> dict:
    C = 0.90
    clean = re.sub(r"^```(?:json)?\s*","",raw_text.strip(),flags=re.M)
    clean = re.sub(r"\s*```$","",clean.strip(),flags=re.M)
    try: parsed = json.loads(clean)
    except:
        m = re.search(r"\{.*\}",clean,re.DOTALL)
        if m:
            try: parsed = json.loads(m.group())
            except: return _empty_out()
        else: return _empty_out()
    out = _empty_out(); out["full_text"]=""; out["page_count"]=0
    def _wrap(src,tgt):
        if not isinstance(src,dict): return
        for k,v in src.items():
            if v is not None and v!="": tgt[k]=_cv(v,C)
    _wrap(parsed.get("header",{}),out["header"])
    _wrap(parsed.get("banking",{}),out["banking"])
    _wrap(parsed.get("totals",{}),out["totals"])
    _wrap(parsed.get("terms",{}),out["terms"])
    _wrap(parsed.get("metadata",{}),out["metadata"])
    pts = parsed.get("parties",{})
    for sub in ["bill_to","ship_to","remit_to"]:
        _wrap(pts.get(sub,{}),out["parties"])
    for prefix,keys in [("bill_to",["name","address","tax_reg"]),("ship_to",["name","address"])]:
        for k in keys:
            if k in out["parties"] and f"{prefix}_{k}" not in out["parties"]:
                out["parties"][f"{prefix}_{k}"] = out["parties"].pop(k)
    for row in parsed.get("line_items",[]):
        if isinstance(row,dict):
            out["line_items"].append({k:_cv(row.get(k,""),C) for k in _LINE_OUT_KEYS})
    for row in parsed.get("tax_lines",[]):
        if isinstance(row,dict):
            out["tax_lines"].append({k:_cv(row.get(k,""),C) for k in _TAX_OUT_KEYS})
    return out

# ═══════════════════════════════════════════════════════════════════
# CONSENSUS ENGINE
# ═══════════════════════════════════════════════════════════════════

WEIGHTS = {"document_understanding":0.45,"gemini":0.40,"vision":0.15}
ENGINE_PRIORITY = ["document_understanding","gemini","vision"]

def _norm(v):
    s = str(v or "").strip().lower()
    return re.sub(r"\s+"," ",re.sub(r"[,\$\s\-–—]+"," ",s)).strip()

def _ev(f): return f["value"] if isinstance(f,dict) and "value" in f else (f or "")
def _ec(f): return float(f["confidence"]) if isinstance(f,dict) and "confidence" in f else 0.5

def _vote(du,gem,vis):
    cands = {"document_understanding":(_ev(du),_ec(du)),"gemini":(_ev(gem),_ec(gem)),"vision":(_ev(vis),_ec(vis))}
    ne = {k:v for k,v in cands.items() if v[0]}
    if not ne: return "",0.0
    vals = list(ne.values()); norms = [_norm(v[0]) for v in vals]; keys = list(ne.keys())
    for i,ni in enumerate(norms):
        if not ni: continue
        if any(j!=i and nj==ni for j,nj in enumerate(norms)):
            wv = vals[i][0]
            for eng in ENGINE_PRIORITY:
                if eng in ne and _norm(ne[eng][0])==ni: wv=ne[eng][0]; break
            return wv, 0.85
    for eng in ENGINE_PRIORITY:
        if eng in ne and ne[eng][0]: return ne[eng]
    return "",0.0

def _merge_sec(du,gem,vis):
    out={}
    for k in set(du)|set(gem)|set(vis):
        val,conf = _vote(du.get(k),gem.get(k),vis.get(k))
        if val: out[k]={"value":val,"confidence":round(conf,4)}
    return out

def _merge_lines(du_l,gem_l,vis_l):
    def _idx(lines):
        idx={}
        for r in lines:
            n=_norm(_ev(r.get("line_number","")))
            if n: idx[n]=r
        return idx
    di,gi,vi = _idx(du_l),_idx(gem_l),_idx(vis_l)
    nums = sorted(set(di)|set(gi)|set(vi),key=lambda x:(len(x),x))
    merged=[]
    for n in nums:
        row={}
        for col in _LINE_OUT_KEYS:
            val,conf = _vote(di.get(n,{}).get(col),gi.get(n,{}).get(col),vi.get(n,{}).get(col))
            row[col]={"value":val,"confidence":round(conf,4)}
        merged.append(row)
    return merged

def _merge_tax(du_t,gem_t,vis_t):
    def _idx(lines):
        idx={}
        for r in lines:
            n=_norm(_ev(r.get("rate_code","")))
            if n: idx[n]=r
        return idx
    di,gi,vi = _idx(du_t),_idx(gem_t),_idx(vis_t)
    codes=list(di)
    for c in list(gi)+list(vi):
        if c not in codes: codes.append(c)
    merged=[]
    for c in codes:
        row={}
        for col in _TAX_OUT_KEYS:
            val,conf = _vote(di.get(c,{}).get(col),gi.get(c,{}).get(col),vi.get(c,{}).get(col))
            row[col]={"value":val,"confidence":round(conf,4)}
        merged.append(row)
    return merged

def build_consensus(du,vis,gem):
    hdr  = _merge_sec(du.get("header",{}),  gem.get("header",{}),  vis.get("header",{}))
    par  = _merge_sec(du.get("parties",{}), gem.get("parties",{}), vis.get("parties",{}))
    ban  = _merge_sec(du.get("banking",{}), gem.get("banking",{}), vis.get("banking",{}))
    tot  = _merge_sec(du.get("totals",{}),  gem.get("totals",{}),  vis.get("totals",{}))
    trm  = _merge_sec(du.get("terms",{}),   gem.get("terms",{}),   vis.get("terms",{}))
    meta = _merge_sec(du.get("metadata",{}),gem.get("metadata",{}),vis.get("metadata",{}))
    li   = _merge_lines(du.get("line_items",[]),gem.get("line_items",[]),vis.get("line_items",[]))
    tx   = _merge_tax(du.get("tax_lines",[]),gem.get("tax_lines",[]),vis.get("tax_lines",[]))

    def _flat(s): return {k:v["value"] for k,v in s.items()}
    def _flat_rows(rows): return [{c:cell["value"] for c,cell in r.items()} for r in rows]
    def _conf_sec(s): return {k:v["confidence"] for k,v in s.items()}

    pf = _flat(par)
    bill_to  = {k.replace("bill_to_",""):v for k,v in pf.items() if k.startswith("bill_to_")}
    ship_to  = {k.replace("ship_to_",""):v for k,v in pf.items() if k.startswith("ship_to_")}
    remit_to = {k:v for k,v in pf.items() if not k.startswith("bill_to_") and not k.startswith("ship_to_")}

    return {
        "header":     _flat(hdr),
        "bill_to":    bill_to,
        "ship_to":    ship_to,
        "remit_to":   remit_to,
        "banking":    _flat(ban),
        "line_items": _flat_rows(li),
        "tax_lines":  _flat_rows(tx),
        "totals":     _flat(tot),
        "terms":      _flat(trm),
        "metadata":   _flat(meta),
        "_confidence":{
            "header":  _conf_sec(hdr),
            "parties": _conf_sec(par),   # flat keys e.g. bill_to_address, ship_to_name
            "banking": _conf_sec(ban),
            "totals":  _conf_sec(tot),
            "terms":   _conf_sec(trm),
            "metadata":_conf_sec(meta),
            "line_items":[{c:cell["confidence"] for c,cell in r.items()} for r in li],
            "tax_lines": [{c:cell["confidence"] for c,cell in r.items()} for r in tx],
        },
    }

# ═══════════════════════════════════════════════════════════════════
# EXTRACTION ENGINES  —  all inline / synchronous, no Object Storage needed
# ═══════════════════════════════════════════════════════════════════

def run_du(clients, pdf_bytes: bytes) -> dict:
    """
    Document Understanding via AnalyzeDocument (inline).
    Automatically splits PDFs exceeding OCI's 5-page inline limit into
    chunks, processes each chunk, then merges the results.
    """
    chunks = split_pdf_chunks(pdf_bytes)
    log.info("[DU] Starting AnalyzeDocument — %d chunk(s)", len(chunks))
    results = []
    for idx, chunk in enumerate(chunks, 1):
        log.info("[DU] Processing chunk %d/%d", idx, len(chunks))
        b64 = base64.b64encode(chunk).decode()
        resp = clients["doc_sync"].analyze_document(
            analyze_document_details=oci.ai_document.models.AnalyzeDocumentDetails(
                document=oci.ai_document.models.InlineDocumentDetails(
                    source="INLINE", data=b64),
                features=[oci.ai_document.models.DocumentKeyValueExtractionFeature(
                    model_id=CUSTOM_MODEL_OCID)],
                compartment_id=COMPARTMENT_ID,
            )
        )
        results.append(parse_du(_sdk_obj_to_dict(resp.data)))
    merged = merge_du_results(results)
    log.info("[DU] Complete — %d line items, %d tax lines",
             len(merged.get("line_items", [])), len(merged.get("tax_lines", [])))
    return merged


def _sdk_obj_to_dict(obj):
    """Recursively convert OCI SDK model objects to plain dicts for parsing."""
    if hasattr(obj, '__dict__'):
        return {k: _sdk_obj_to_dict(v) for k, v in obj.__dict__.items()
                if not k.startswith('_')}
    elif isinstance(obj, list):
        return [_sdk_obj_to_dict(i) for i in obj]
    else:
        return obj


def run_vision(clients, pdf_bytes: bytes) -> dict:
    """
    Vision OCR via AnalyzeDocument (inline).
    Automatically splits PDFs exceeding OCI's 5-page inline limit,
    processes each chunk, then merges the OCR text results.
    """
    chunks = split_pdf_chunks(pdf_bytes)
    log.info("[Vision] Starting AnalyzeDocument — %d chunk(s)", len(chunks))
    results = []
    for idx, chunk in enumerate(chunks, 1):
        log.info("[Vision] Processing chunk %d/%d", idx, len(chunks))
        b64 = base64.b64encode(chunk).decode()
        resp = clients["vision"].analyze_document(
            analyze_document_details=oci.ai_vision.models.AnalyzeDocumentDetails(
                document=oci.ai_vision.models.InlineDocumentDetails(
                    source="INLINE", data=b64),
                features=[oci.ai_vision.models.DocumentTextDetectionFeature()],
                compartment_id=COMPARTMENT_ID,
                language="ENG",
            )
        )
        results.append(parse_vision(resp.data, fallback=True))
    merged = merge_vision_results(results)
    log.info("[Vision] Complete — %d pages, %d OCR chars",
             merged.get("page_count", 0), len(merged.get("full_text", "")))
    return merged


def run_vision_image(clients, image_bytes: bytes) -> dict:
    """
    Vision OCR for a raw image file (JPG, PNG, TIFF, etc.).
    OCI Vision's AnalyzeImage endpoint accepts image bytes directly —
    no PDF conversion needed.  Used as a fallback when image_to_pdf() fails.
    """
    log.info("[Vision/Image] Sending raw image bytes to OCI Vision AnalyzeImage")
    b64 = base64.b64encode(image_bytes).decode()
    resp = clients["vision"].analyze_image(
        analyze_image_details=oci.ai_vision.models.AnalyzeImageDetails(
            image=oci.ai_vision.models.InlineImageDetails(
                source="INLINE", data=b64),
            features=[oci.ai_vision.models.ImageTextDetectionFeature()],
        )
    )
    # AnalyzeImage returns a flat response (no pages); re-use parse_vision in fallback mode
    out = _empty_out()
    out["full_text"] = ""
    out["page_count"] = 1
    lines = []
    for ln in getattr(resp.data, "image_text", None) and \
              getattr(resp.data.image_text, "lines", []) or []:
        t = getattr(ln, "text", "") or ""
        if t.strip():
            lines.append(t.strip())
    out["full_text"] = "\n".join(lines)
    log.info("[Vision/Image] Complete — %d OCR chars", len(out["full_text"]))
    return out


def run_gemini(clients, pdf_bytes, du_fields, vis_fields) -> dict:
    log.info("[Gemini] Calling Gemini 2.5 Pro")
    sys_text = (
        "You are an expert Oracle Fusion Receivables invoice extraction engine. "
        "You receive: (1) structured Document Understanding fields, (2) raw OCR text. "
        "Validate DU values, fill gaps, correct OCR errors, extract ALL line items (7 cols each), "
        "ALL tax rows, ensure totals are mathematically consistent. "
        "Return ONLY valid JSON — no markdown, no preamble:\n"
        '{"header":{...},"parties":{"bill_to":{...},"ship_to":{...},"remit_to":{...}},'
        '"banking":{...},"line_items":[{"line_number":...,"description":...,"uom":...,'
        '"quantity":...,"unit_price":...,"discount_pct":...,"amount":...}],'
        '"tax_lines":[{"regime":...,"name":...,"rate_code":...,"taxable_amount":...,'
        '"rate_pct":...,"tax_amount":...}],'
        '"totals":{"subtotal":...,"discount_total":...,"taxable_amount":...,"total_tax":...,"invoice_total":...},'
        '"terms":{"payment_terms":...,"late_fee":...,"dispute_email":...,"invoice_status":...},'
        '"metadata":{"printed_date":...,"transaction_source":...,"authorized_by":...}}'
    )
    user_text = (
        f"## Document Understanding Fields\n```json\n{json.dumps(du_fields,indent=2,default=str)[:5000]}\n```\n\n"
        f"## Vision OCR Text\n```\n{vis_fields.get('full_text','')[:6000]}\n```\n\nProduce the final JSON."
    )
    # system_message is not a valid kwarg in older SDK versions — send it as the
    # first message with role SYSTEM instead.
    chat_req = oci.generative_ai_inference.models.GenericChatRequest(
        api_format=oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC,
        messages=[
            oci.generative_ai_inference.models.SystemMessage(
                role="SYSTEM",
                content=[oci.generative_ai_inference.models.TextContent(
                    type="TEXT", text=sys_text)],
            ),
            oci.generative_ai_inference.models.UserMessage(
                role="USER",
                content=[oci.generative_ai_inference.models.TextContent(
                    type="TEXT", text=user_text)],
            ),
        ],
        max_tokens=8192, temperature=0.0, is_stream=False,
    )
    resp = clients["genai"].chat(oci.generative_ai_inference.models.ChatDetails(
        compartment_id=COMPARTMENT_ID,
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=GEMINI_MODEL),
        chat_request=chat_req,
    ))
    _record_metric("gemini", "invoice_extraction",
                   getattr(resp.data.chat_response, "usage", None))
    return parse_gemini(resp.data.chat_response.choices[0].message.content[0].text)


# ═══════════════════════════════════════════════════════════════════
# SCHEMA GENERATION  —  Gemini + Grok two-stage schema induction
# ═══════════════════════════════════════════════════════════════════

_VALID_SCHEMA_TYPES = {"string", "date", "currency", "number"}

_SG_GEMINI_SYSTEM = """\
You are an expert document schema analyst for OCI Document Understanding model training.
Analyse the provided document and produce a complete key-value extraction schema.

NAMING RULES (critical — these keys become OCI DU label identifiers):
  • Use PascalCase for every key  (e.g. InvoiceNumber, BillToAddress, LineItemAmount)
  • Use these exact names for common fields — do not abbreviate or expand them:
    PurchaseOrderNumber, BillToTaxRegistrationNumber, LineNumber, SubTotal,
    BankRoutingNumber, SWIFTCode, LineItemUOM, LineItemDiscountPercent,
    TaxName, TaxRateCode, TaxLineTaxableAmount, TaxLineRate, TaxLineAmount,
    TotalTaxAmount, InvoiceTotal, LatePaymentFee, DisputeContactEmail

DESCRIPTION RULES — every description must contain ALL of:
  1. Exact printed label nearest to the field  (e.g. "Labeled 'Invoice Number' in …")
  2. Spatial location on the page  (e.g. "upper-right Invoice Details panel", "page footer")
  3. Concrete format example  (e.g. "Format: DD-Mon-YYYY, e.g. 05-May-2026")
  4. Disambiguation: "Do not extract X or Y" for any field easily confused with another

DATATYPE rules:
  • Values that include a % sign (e.g. "10.00%", "9.25%") → dataType "string", NOT "number"
  • Monetary amounts → dataType "currency"
  • Pure counts / quantities → dataType "number"
  • Dates → dataType "date"
  • Everything else → dataType "string"

TABLE COLUMN RULES:
  • The first column of each table is the anchor key; its description does NOT reference another key
  • Every other column in the same table MUST end with:
    "Return as a list preserving table order, with the same length and ordering as [AnchorKey]"

TOTALS / CURRENCY FIELDS:
  • Include the currency prefix in format examples  (e.g. "USD 126,615.00" not "126615.00")

DOCUMENT ZONES — you MUST produce at least one field from EVERY zone below:
  1. Header / Invoice Details panel  (invoice number, dates, terms, currency, BU, type, PO, salesperson)
  2. Bill-To block  (name, full address, tax registration number)
  3. Ship-To block  (name, full address, attention line if present)
  4. Remittance Information  (remit-to name, address, bank name, account, routing, SWIFT, wire ref)
  5. Line Items table  (one field per column: #, description, UOM, qty, unit price, discount, amount)
  6. Tax Details table  (one field per column: regime, name, rate code, taxable amount, rate %, tax amount)
  7. Totals summary  (subtotal, discount total, taxable amount, tax amount, invoice total)
  8. Notes & Payment Terms section  (LatePaymentFee, InvoiceStatus, DisputeContactEmail)
  9. Page footer  (PrintedDate, TransactionSource, AuthorizedBy)

Do NOT include free-form paragraph text or boilerplate legal notices.

Return ONLY a valid JSON array — no markdown fences, no explanation:
[{"key":"...","dataType":"...","description":"..."}, ...]"""

_SG_GROK_SYSTEM = """\
You are a senior data-labelling engineer finalising an OCI Document Understanding schema.
Apply every rule below without exception.

1. KEY NAMES
   a. All keys must be PascalCase. Rename any camelCase or abbreviated key.
   b. DO NOT rename a key that is already valid PascalCase — renaming creates duplicates
      in the merge step. Only rename if the key is camelCase or genuinely ambiguous.
   c. Use these exact canonical names — correct any deviation:
        PurchaseOrderNumber      (not PoNumber, PONumber, PurchaseOrder)
        BillToTaxRegistrationNumber (not BillToTaxId, BillToTaxNumber)
        LineNumber               (not LineItemNumber, ItemNumber)
        SubTotal                 (not Subtotal — capital T)
        BankRoutingNumber        (not ABARoutingNumber, AbaRoutingNumber, RoutingNumber)
        SWIFTCode                (not SwiftCode, SwiftBIC)
        LineItemUOM              (not LineItemUnitOfMeasure, UOM)
        LineItemDiscountPercent  (not LineItemDiscountPercentage)
        TaxName                  (not TaxDetailName)
        TaxRateCode              (not TaxDetailRateCode)
        TaxLineTaxableAmount     (not TaxDetailTaxableAmount)
        TaxLineRate              (not TaxDetailRatePercentage, TaxRate)
        TaxLineAmount            (not TaxDetailAmount)
        TotalTaxAmount           (not TaxTotal)
        InvoiceTotal             (not TotalAmountDue)
        LatePaymentFee           (not LateFee, LatePaymentPenalty)
        DisputeContactEmail      (not DisputeProcess — extract ONLY the email address substring)

2. DATATYPES
   • Values containing a % sign → dataType "string" (not "number")
   • Monetary values → "currency"
   • Pure counts → "number"
   • Dates → "date"

3. DESCRIPTIONS — every entry must have:
   a. Exact printed label  ("Labeled 'X' in the Y section")
   b. Spatial location
   c. Format example with currency prefix  ("USD 126,615.00" not "126615.00")
   d. "Do not extract X" disambiguation

4. TABLE ANCHOR RULE
   • LineNumber is the anchor for the line items table — its description must NOT say
     "same length as LineNumber" (self-reference bug)
   • TaxRegime is the anchor for the tax details table — same rule applies
   • All non-anchor columns must end with the "same length and ordering as [Anchor]" clause

5. MISSING ZONES — ADD any of these that are absent:
   • Notes & Payment Terms:  LatePaymentFee, InvoiceStatus, DisputeContactEmail
   • Page footer:            PrintedDate, TransactionSource, AuthorizedBy

6. REMOVE noise: free-form Notes paragraphs, boilerplate LegalNotice

7. ORDER: header → parties → banking → line items → tax lines → totals → terms → footer

Return ONLY a valid JSON array — no markdown, no commentary:
[{"key":"...","dataType":"...","description":"..."}, ...]"""

_SG_GEMINI_USER = (
    "## Vision OCR Text (all pages including footer)\n```\n{ocr}\n```\n\n"
    "{pdf_note}\n\n"
    "Scan EVERY line including the page footer (Printed:, Transaction Source:, Authorized by:) "
    "and the 'NOTES & PAYMENT TERMS' section (Late Fee:, Invoice Status:, Dispute Process:) "
    "before generating.\n\n"
    "Generate the complete extraction schema JSON array now."
)

_SG_GROK_USER = (
    "## Original Document Text (all pages)\n```\n{ocr}\n```\n\n"
    "## Draft Schema ({n} fields)\n```json\n{draft}\n```\n\n"
    "Verify the draft covers: Notes & Payment Terms (LatePaymentFee, InvoiceStatus, "
    "DisputeContactEmail) and page footer (PrintedDate, TransactionSource, AuthorizedBy). "
    "Apply canonical key names. Fix dataTypes for % fields. "
    "DO NOT rename keys that are already valid PascalCase. "
    "Produce the final validated schema array now."
)


def _sg_deduplicate(schema: list[dict]) -> list[dict]:
    """
    Remove semantic duplicates produced by a single model or by the Gemini→Grok rename
    cycle where both key versions end up in the merged output.

    Three passes (each keeps the entry with the longer description):
      1. Case-insensitive key match     — ABARoutingNumber vs AbaRoutingNumber
      2. 'Labeled X' anchor             — fields that use "Labeled 'X' in …" form
      3. Table-column anchor            — fields that use "from the 'X' column in the 'Y' table"
                                          catches TaxRegime vs TaxLineRegime etc.
    """
    def _keep(existing: dict, challenger: dict) -> dict:
        return challenger if len(challenger["description"]) > len(existing["description"]) else existing

    def _labeled(desc: str) -> str | None:
        m = re.search(r"[Ll]abeled\s+'([^']+)'", desc)
        return m.group(1).strip().lower() if m else None

    def _col_in_table(desc: str) -> str | None:
        # matches: from the 'X' column in the 'Y' table
        m = re.search(r"from the '([^']+)' column in the '([^']+)' table", desc, re.I)
        if m:
            return f"{m.group(2).lower()}|{m.group(1).lower()}"
        # also: from the 'X' column (without table name)
        m2 = re.search(r"from the '([^']+)' column", desc, re.I)
        return m2.group(1).lower() if m2 else None

    # Pass 1: case-insensitive key
    seen_key: dict[str, int] = {}
    p1: list[dict] = []
    for e in schema:
        kl = e["key"].lower()
        if kl in seen_key:
            p1[seen_key[kl]] = _keep(p1[seen_key[kl]], e)
        else:
            seen_key[kl] = len(p1); p1.append(e)

    # Pass 2: Labeled 'X' anchor
    seen_lbl: dict[str, int] = {}
    p2: list[dict] = []
    for e in p1:
        lbl = _labeled(e.get("description", ""))
        if lbl and lbl in seen_lbl:
            p2[seen_lbl[lbl]] = _keep(p2[seen_lbl[lbl]], e)
        else:
            if lbl: seen_lbl[lbl] = len(p2)
            p2.append(e)

    # Pass 3: table-column anchor
    seen_col: dict[str, int] = {}
    output: list[dict] = []
    for e in p2:
        col = _col_in_table(e.get("description", ""))
        if col and col in seen_col:
            output[seen_col[col]] = _keep(output[seen_col[col]], e)
        else:
            if col: seen_col[col] = len(output)
            output.append(e)

    removed = len(schema) - len(output)
    if removed:
        log.info("[SG/Dedup] Removed %d semantic duplicate(s)", removed)
    return output


def _sg_parse(raw: str, source: str) -> list[dict]:
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.M)
    clean = re.sub(r"\s*```\s*$", "", clean.strip(), flags=re.M)
    parsed = None
    for candidate in [clean, None]:
        if candidate is None:
            m = re.search(r"\[.*\]", clean, re.DOTALL)
            candidate = m.group() if m else ""
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            continue
    if not isinstance(parsed, list):
        log.error("[SG/%s] No valid JSON array found", source)
        return []
    out = []
    for e in parsed:
        if not isinstance(e, dict):
            continue
        key  = str(e.get("key", "")).strip()
        dt   = str(e.get("dataType", "string")).strip()
        desc = str(e.get("description", "")).strip()
        if not key or not desc:
            continue
        if dt not in _VALID_SCHEMA_TYPES:
            dt = "string"
        out.append({"key": key, "dataType": dt, "description": desc})
    log.info("[SG/%s] %d fields parsed", source, len(out))
    return out


def _sg_genai_call(clients, model_id: str, system: str, user_parts: list,
                   operation: str = "schema") -> str:
    M = oci.generative_ai_inference.models
    req = M.GenericChatRequest(
        api_format=M.BaseChatRequest.API_FORMAT_GENERIC,
        messages=[
            M.SystemMessage(role="SYSTEM",
                content=[M.TextContent(type="TEXT", text=system)]),
            M.UserMessage(role="USER", content=user_parts),
        ],
        max_tokens=8192, temperature=0.0, is_stream=False,
    )
    resp = clients["genai"].chat(M.ChatDetails(
        compartment_id=COMPARTMENT_ID,
        serving_mode=M.OnDemandServingMode(model_id=model_id),
        chat_request=req,
    ))
    engine = "grok" if model_id == GROK_MODEL else "gemini"
    _record_metric(engine, operation, getattr(resp.data.chat_response, "usage", None))
    return resp.data.chat_response.choices[0].message.content[0].text


def sg_run_gemini(clients, ocr_text: str, pdf_bytes: bytes) -> list[dict]:
    log.info("[SG/Gemini] Inducing schema draft…")
    M = oci.generative_ai_inference.models
    parts = []
    pdf_note = ""
    try:
        b64 = base64.b64encode(pdf_bytes).decode()
        parts.append(M.ImageContent(
            type="IMAGE",
            image_url=M.ImageUrl(url=f"data:application/pdf;base64,{b64}"),
        ))
        pdf_note = "PDF attached inline — use visual layout for precise field locations."
        log.info("[SG/Gemini] PDF inline (%d KB)", len(pdf_bytes) // 1024)
    except Exception as exc:
        pdf_note = "No inline PDF — infer layout from OCR text patterns."
        log.warning("[SG/Gemini] Inline PDF skipped (%s)", exc)
    parts.append(M.TextContent(type="TEXT",
        text=_SG_GEMINI_USER.format(ocr=ocr_text[:8000], pdf_note=pdf_note)))
    raw = _sg_genai_call(clients, GEMINI_MODEL, _SG_GEMINI_SYSTEM, parts, "schema_induction")
    return _sg_parse(raw, "Gemini")


def sg_run_grok(clients, ocr_text: str, draft: list[dict]) -> list[dict]:
    log.info("[SG/Grok] Reinforcing schema (%d fields)…", len(draft))
    M = oci.generative_ai_inference.models
    user_text = _SG_GROK_USER.format(
        ocr=ocr_text[:8000], n=len(draft), draft=json.dumps(draft, indent=2))
    raw = _sg_genai_call(clients, GROK_MODEL, _SG_GROK_SYSTEM,
                         [M.TextContent(type="TEXT", text=user_text)],
                         "schema_reinforcement")
    return _sg_parse(raw, "Grok")


def sg_merge(gemini: list[dict], grok: list[dict]) -> list[dict]:
    grok_keys = {e["key"] for e in grok}
    merged = list(grok)
    for e in gemini:
        if e["key"] not in grok_keys:
            merged.append(e)
            log.info("[SG/Merge] Gemini-only field kept: %s", e["key"])
    log.info("[SG/Merge] %d total fields", len(merged))
    return merged


# ═══════════════════════════════════════════════════════════════════
# TOKEN METRICS  —  session-level usage tracking across all LLM calls
# ═══════════════════════════════════════════════════════════════════

_metrics_lock: threading.Lock = threading.Lock()
_metrics_log:  list[dict]     = []


def _record_metric(engine: str, operation: str, usage) -> None:
    """Extract token counts from an OCI GenAI usage object and append to the log."""
    if usage is None:
        return
    try:
        prompt     = int(getattr(usage, "prompt_tokens",     0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total      = int(getattr(usage, "total_tokens",      0) or 0)
        reasoning  = 0
        details    = getattr(usage, "completion_tokens_details", None)
        if details:
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
        entry = {
            "ts":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engine":     engine,
            "operation":  operation,
            "prompt":     prompt,
            "completion": completion,
            "reasoning":  reasoning,
            "total":      total,
        }
        with _metrics_lock:
            _metrics_log.append(entry)
        log.info("[Metrics] %s/%s — prompt:%d completion:%d reasoning:%d total:%d",
                 engine, operation, prompt, completion, reasoning, total)
    except Exception as exc:
        log.warning("[Metrics] Could not record usage: %s", exc)


# ═══════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════

_HERE      = Path(__file__).parent
_HTML_FILE = _HERE / "trifectai.html"
_HTML_CACHE: Optional[str] = None

def _get_html() -> str:
    global _HTML_CACHE
    if _HTML_CACHE is None:
        _HTML_CACHE = _HTML_FILE.read_text(encoding="utf-8")
    return _HTML_CACHE

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

CORS(app)

# ── OIDC via Authlib ────────────────────────────────────────────────
oauth = OAuth(app)
oauth.register(
    "ocidomain",
    client_id=OIDC_CLIENT_ID,
    client_secret=OIDC_CLIENT_SECRET,
    server_metadata_url=f"{OIDC_DOMAIN_URL}/.well-known/openid-configuration",
    client_kwargs={"scope": "openid"},
)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            if request.path.startswith("/extract"):
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Auth routes ─────────────────────────────────────────────────────
@app.route("/login")
def login():
    callback_uri = url_for("callback", _external=True)
    return oauth.ocidomain.authorize_redirect(callback_uri, prompt="login")

@app.route("/callback")
def callback():
    token = oauth.ocidomain.authorize_access_token()
    info  = token.get("userinfo", {})
    session["user"]         = info.get("sub", "")
    session["display_name"] = info.get("user_displayname") or info.get("name") or info.get("sub", "User")
    session["email"]        = info.get("sub", "")
    session["user_ocid"]    = info.get("user_ocid", "")
    session["id_token"]     = token.get("id_token", "")
    return redirect(url_for("serve_ui"))

@app.route("/logout")
def logout():
    for key in ("user", "display_name", "email", "user_ocid", "id_token"):
        session.pop(key, None)
    return redirect(url_for("signed_out"))

@app.route("/signed-out")
def signed_out():
    return Response(_get_html(), mimetype="text/html")

@app.route("/me")
@login_required
def me():
    return jsonify({
        "display_name": session.get("display_name", "User"),
        "email":        session.get("email", ""),
        "user_ocid":    session.get("user_ocid", ""),
    })

@app.route("/")
@login_required
def serve_ui():
    if not _HTML_FILE.exists():
        return jsonify({"error": "trifectai.html not found"}), 404
    return Response(_get_html(), mimetype="text/html")

@app.route("/extract", methods=["POST"])
@login_required
def extract():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]

    # ── Validate file format ─────────────────────────────────────────
    ext = Path(f.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. "
                     f"Accepted formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        }), 400

    raw_bytes = f.read()
    file_name = f.filename
    is_image  = ext in IMAGE_EXTENSIONS
    start     = time.time()

    # ── Convert image → PDF so all three engines get the same bytes ──
    # If Pillow isn't installed or the conversion fails, we keep raw_bytes
    # and skip DU (which requires a PDF) while Vision/Gemini still run.
    pdf_bytes       = raw_bytes   # what we hand to DU + Gemini
    can_run_du      = not is_image  # DU requires a real PDF
    vision_is_image = False         # flag so we know which Vision call to make

    if is_image:
        try:
            pdf_bytes  = image_to_pdf(raw_bytes)
            can_run_du = True           # conversion succeeded — DU can run too
            log.info("[Extract] Image converted to PDF for all engines")
        except ImportError:
            log.warning("[Extract] Pillow not installed — DU skipped, using raw image for Vision")
            vision_is_image = True
        except Exception as e:
            log.warning("[Extract] image_to_pdf failed (%s) — DU skipped, raw image for Vision", e)
            vision_is_image = True

    try:
        clients = _clients()
    except Exception as e:
        return jsonify({"error": f"OCI auth failed: {e}. Check oci_key.txt and config."}), 500

    du_result = vis_result = gem_result = _empty_out()
    engine_status = {}

    # ── Run DU + Vision concurrently ────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # Only submit DU if we have a real PDF (native or converted)
        fu_du = pool.submit(run_du, clients, pdf_bytes) if can_run_du else None

        # Vision: use the raw-image endpoint if PDF conversion failed,
        # otherwise use the normal PDF path
        if vision_is_image:
            fu_vis = pool.submit(run_vision_image, clients, raw_bytes)
        else:
            fu_vis = pool.submit(run_vision, clients, pdf_bytes)

        if fu_du:
            try:
                du_result = fu_du.result(timeout=300)
                engine_status["document_understanding"] = "ok"
            except Exception as e:
                log.error("[DU] %s", e)
                engine_status["document_understanding"] = str(e)
        else:
            engine_status["document_understanding"] = "skipped (image input, Pillow not available)"

        try:
            vis_result = fu_vis.result(timeout=120)
            engine_status["vision"] = "ok"
        except Exception as e:
            log.error("[Vision] %s", e)
            engine_status["vision"] = str(e)

    # ── Run Gemini with DU + Vision context ─────────────────────────
    try:
        gem_result = run_gemini(clients, pdf_bytes, du_result, vis_result)
        engine_status["gemini"] = "ok"
    except Exception as e:
        log.error("[Gemini] %s", e)
        engine_status["gemini"] = str(e)

    final = build_consensus(du_result, vis_result, gem_result)
    final["_extraction_meta"] = {
        "source_file":  file_name,
        "file_type":    "image" if is_image else "pdf",
        "elapsed_sec":  round(time.time() - start, 2),
        "engines":      engine_status,
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return jsonify(final)

@app.route("/generate-schema", methods=["POST"])
@login_required
def generate_schema_route():
    files = request.files.getlist("files")
    pdf_files = [(f.filename, f.read()) for f in files
                 if f.filename and f.filename.lower().endswith(".pdf")]
    if not pdf_files:
        return jsonify({"error": "Upload at least one PDF file"}), 400

    mode  = request.form.get("mode", "full")   # "full" or "gemini_only"
    start = time.time()
    stages = {"ocr": "pending", "gemini": "pending",
              "grok": ("pending" if mode == "full" else "skipped")}

    try:
        clients = _clients()
    except Exception as e:
        return jsonify({"error": f"OCI auth failed: {e}"}), 500

    # ── OCR all PDFs ─────────────────────────────────────────────
    ocr_parts: list[str] = []
    primary_pdf: bytes = b""
    try:
        for name, pdf_bytes in pdf_files:
            if not primary_pdf:
                primary_pdf = pdf_bytes
            vis = run_vision(clients, pdf_bytes)
            ocr_parts.append(f"=== {name} ===\n{vis.get('full_text', '')}")
        combined_ocr = "\n\n".join(ocr_parts)
        stages["ocr"] = "ok"
    except Exception as e:
        log.error("[SG] OCR failed: %s", e)
        return jsonify({"error": f"OCR failed: {e}", "stages": stages}), 500

    # ── Gemini: induce draft schema ───────────────────────────────
    try:
        gemini_schema = _sg_deduplicate(sg_run_gemini(clients, combined_ocr, primary_pdf))
        if not gemini_schema:
            raise ValueError("Gemini returned no schema fields")
        stages["gemini"] = "ok"
    except Exception as e:
        log.error("[SG] Gemini failed: %s", e)
        return jsonify({"error": f"Schema induction failed: {e}", "stages": stages}), 500

    final_schema = gemini_schema

    # ── Grok: reinforce and validate ─────────────────────────────
    if mode == "full":
        try:
            grok_schema = sg_run_grok(clients, combined_ocr, gemini_schema)
            if grok_schema:
                final_schema = _sg_deduplicate(sg_merge(gemini_schema, grok_schema))
                stages["grok"] = "ok"
            else:
                stages["grok"] = "no_output"
        except Exception as e:
            log.error("[SG] Grok failed: %s", e)
            stages["grok"] = str(e)[:120]

    return jsonify({
        "schema": final_schema,
        "meta": {
            "files":       [n for n, _ in pdf_files],
            "field_count": len(final_schema),
            "mode":        mode,
            "elapsed_sec": round(time.time() - start, 2),
            "stages":      stages,
        },
    })


@app.route("/metrics")
@login_required
def get_metrics():
    with _metrics_lock:
        log_copy = list(_metrics_log)

    totals = {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0}
    by_engine: dict[str, dict] = {}
    for e in log_copy:
        for k in totals:
            totals[k] += e.get(k, 0)
        eng = e["engine"]
        if eng not in by_engine:
            by_engine[eng] = {"calls": 0, "prompt": 0, "completion": 0,
                               "reasoning": 0, "total": 0}
        by_engine[eng]["calls"] += 1
        for k in ("prompt", "completion", "reasoning", "total"):
            by_engine[eng][k] += e.get(k, 0)

    return jsonify({
        "totals":     totals,
        "by_engine":  by_engine,
        "log":        list(reversed(log_copy)),   # newest first
        "call_count": len(log_copy),
    })


@app.route("/metrics/reset", methods=["POST"])
@login_required
def reset_metrics():
    with _metrics_lock:
        _metrics_log.clear()
    log.info("[Metrics] Session metrics cleared")
    return jsonify({"status": "cleared"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "trifectai"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)