# utils_ocr.py
import io
import base64
import tempfile
import os
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from groq import Groq
import camelot
import re


# ---------------------------------------------------------
# Detect if PDF is vector (Camelot can parse)
# ---------------------------------------------------------
def is_vector_pdf(pdf_path: str) -> bool:
    """
    Heuristic:
    - Try Camelot lattice on page 1
    - Then Camelot stream
    If ANY returns a table → PDF is vector-based.
    """
    try:
        tables = camelot.read_pdf(pdf_path, pages="1", flavor="lattice")
        if len(tables) > 0:
            return True

        tables = camelot.read_pdf(pdf_path, pages="1", flavor="stream")
        return len(tables) > 0

    except Exception:
        return False


# ---------------------------------------------------------
# Camelot Table Extraction
# ---------------------------------------------------------
def extract_tables_with_camelot(pdf_path: str, max_pages: int = 50):
    """
    Try both lattice and stream.
    Returns: list of DataFrames
    """
    dataframes = []
    pages = f"1-{max_pages}"

    for flavor in ["lattice", "stream"]:
        try:
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor)
            for t in tables:
                df = t.df.replace("\n", " ", regex=True)
                dataframes.append(df)

            if len(dataframes) > 0:
                break

        except Exception:
            continue

    return dataframes


# ---------------------------------------------------------
# PDF → Images (for OCR)
# ---------------------------------------------------------
def convert_pdf_to_images(pdf_path: str, dpi: int = 300, max_pages: int = 50, poppler_path: str = None):
    """
    Convert PDF to PIL images.
    """
    poppler_path_final = poppler_path or os.getenv("POPPLER_PATH")
    pages = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=1,
        last_page=max_pages,
        poppler_path=poppler_path_final
    )
    return pages

# ---------------------------------------------------------
# Convert PIL → base64 PNG
# ---------------------------------------------------------
def _pil_to_base64_png(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------
# OCR — Mistral Vision via Groq
# ---------------------------------------------------------
def extract_tables_with_mistral_ocr(images, api_key: str, model_name: str, table_hint: str):
    """
    Full OCR extraction pipeline with table filtering.

    Returns:
      dict { page_index : DataFrame }
    """
    client = Groq(api_key=api_key)
    results = {}

    # ============================
    # SYSTEM PROMPT
    # ============================
    system_prompt = """
    You are an expert OCR system specialized in extracting ONLY the main
    multi-page Visit Schedule / Schedule of Assessments / Document Attribute Matrix table.

    RULES:
    - Extract ONLY the largest, widest table with Visit columns.
    - Must contain visit keywords like:
      V1, V2, V3, Visit 1, Visit 2, Day 0, Day 14, Day 28, Randomisation, Screening.
    - IGNORE unrelated tables (demographics, title boxes, key tables, summaries).
    - Do NOT hallucinate. Missing cells = "".
    - Output ONLY strict JSON:
        {
          "table_present": true/false,
          "headers": [...],
          "rows": [...],
          "note": ""
        }
    """

    # Loop page-by-page
    for idx, pil_img in enumerate(images):
        b64 = _pil_to_base64_png(pil_img)

        user_prompt = (
            f"Page {idx+1} of the document.\n"
            f"Extract ONLY the Visit Schedule / Assessment Matrix.\n"
            f"Do NOT extract unrelated tables.\n"
            f"{table_hint}\n"
            "Return strict JSON only."
        )

        message_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ]

        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message_content}
                ]
            )

            text = resp.choices[0].message.content.strip()

            # Strip JSON fences
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = "\n".join(text.splitlines()[1:]).strip()

            # Parse JSON
            import json
            payload = json.loads(text)

            # Skip page if no table
            if not payload.get("table_present", False):
                results[idx] = pd.DataFrame()
                continue

            headers = payload.get("headers", [])
            rows = payload.get("rows", [])

            # Flatten multi-level headers
            flat_headers = [
                " | ".join(h) if isinstance(h, list) else str(h)
                for h in headers
            ]

            # Normalize row lengths
            normalized_rows = []
            for row in rows:
                row = ["" if x is None else str(x) for x in row]

                if len(row) < len(flat_headers):
                    row += [""] * (len(flat_headers) - len(row))
                elif len(row) > len(flat_headers):
                    row = row[:len(flat_headers)]

                normalized_rows.append(row)

            df = pd.DataFrame(normalized_rows, columns=flat_headers)

            # ------------------------------------
            # FILTER 1: Table Density (skip sparse junk)
            # ------------------------------------
            df_str = df.astype(str)
            total = df_str.size
            non_empty = df_str.apply(lambda col: col.str.strip() != "").sum().sum()
            density = non_empty / total if total > 0 else 0

            if density < 0.05:  # <5% filled = probably noise
                results[idx] = pd.DataFrame()
                continue

            # ------------------------------------
            # FILTER 2: Column usefulness check
            # ------------------------------------
            useful_cols = (df_str.apply(lambda col: col.str.strip().ne("").sum()) > 0).sum()
            if useful_cols < 4:
                results[idx] = pd.DataFrame()
                continue

            # ------------------------------------
            # FILTER 3: Visit pattern detection
            # ------------------------------------
            header_text = " ".join(flat_headers).lower()
            visit_patterns = [
                r"v\d+",           # V1, V2...
                r"visit\s*\d+",    # Visit 1
                r"day\s*\d+",      # Day 0, Day 14...
                r"random",         # Randomisation / Randomization
                r"screen"          # Screening
            ]

            if not any(re.search(p, header_text) for p in visit_patterns):
                results[idx] = pd.DataFrame()
                continue

            # Accept final table
            results[idx] = df

        except Exception:
            results[idx] = pd.DataFrame()

    return results


# ================================
# BLUE COLOR THRESHOLDS
# ================================
BLUE_MIN_B = 0.60
BLUE_MAX_R = 0.40
BLUE_MAX_G = 0.40

# Character grouping
LINE_TOL = 2.0
TOKEN_GAP_MAX = 2.5

# Meaning overrides dictionary
MEANING_OVERRIDES = {
    "a": "ICF: Ensure the patient has signed the informed consent form if not already done in Run-in.",
    "b": "Endoscopy: Screening colonoscopy/sigmoidoscopy with biopsies; Visit 8 flexible sigmoidoscopy, all centrally read.",
    "c": "Timing: Screening endoscopy must occur between Day −35 and Day −6 before randomisation.",
    "d": "Diary Scores: Stool frequency + rectal bleeding averaged from 3-5 days; excluded after prep and on endoscopy days.",
    "e": "PK Sampling (China): Timepoints around doses 1–6, extending up to Day 105.",
    "f": "Biomarkers: ESR, CRP, counts (local); IL-6, IL-6/sIL-6R, calprotectin (central); fasting at Visits 2 & 8.",
    "g": "Clinical Labs: Haematology, chemistry, coagulation, urinalysis done between Day −5 and Day −1.",
    "h": "Pregnancy test: Serum at Visit 1; Urine at Visits 8 and 9.",
    "i": "Physical exam at Visits 1 & 9; body weight only at Visits 1.1, 4, 6, 8.",
    "j": "Vital signs: BP (after ≥3 min seated), pulse, RR, temperature.",
    "k": "IMP administration is last procedure; infusion duration = 2 hours."
}


# ------------------------------------------------------------
# PDF UTILITIES FOR BLUE-TEXT NOTATION EXTRACTION
# ------------------------------------------------------------

def sanitize_filename(name):
    name = re.sub(r'[^A-Za-z0-9_]+', '_', name)
    return name.strip("_").lower() or "table"


def _to_rgb(color):
    """
    Converts PDF color values into RGB in range [0,1].
    Handles:
      - grayscale
      - RGB 0-1 or 0-255
      - CMYK
    """
    if color is None:
        return None

    # grayscale: single number
    if isinstance(color, (int, float)):
        v = float(color)
        if v > 1.0:
            v /= 255.0
        return (v, v, v)

    # tuple/list
    if isinstance(color, (list, tuple)):
        vals = tuple(float(x) for x in color)

        # RGB
        if len(vals) == 3:
            r, g, b = vals
            if max(vals) > 1.0:
                r /= 255.0
                g /= 255.0
                b /= 255.0
            return (r, g, b)

        # CMYK
        if len(vals) == 4:
            c, m, y, k = vals
            r = 1 - min(1.0, c + k)
            g = 1 - min(1.0, m + k)
            b = 1 - min(1.0, y + k)
            return (r, g, b)

    return None


def _is_blue(rgb):
    """Return True if RGB qualifies as BLUE based on thresholds."""
    if rgb is None:
        return False

    r, g, b = rgb
    return (b >= BLUE_MIN_B) and (r <= BLUE_MAX_R) and (g <= BLUE_MAX_G)


def _line_key(bottom, tol=LINE_TOL):
    """Group characters into same line bucket."""
    return int(round(bottom / tol))


def _char_center_bottom_coords(ch, page_height):
    """
    Returns center x and bottom-aligned y coordinate of a character.
    Required for grouping text into proper reading order.
    """
    xc = 0.5 * (ch.get("x0", 0.0) + ch.get("x1", 0.0))
    yc_top = 0.5 * (ch.get("top", 0.0) + ch.get("bottom", 0.0))
    yc_bottom = page_height - yc_top
    return xc, yc_bottom