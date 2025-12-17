# utils_ocr.py
import pandas as pd
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
        tables = camelot.read_pdf(pdf_path, pages="1", flavor="lattice",flag_size=True)
        if len(tables) > 0:
            return True

        tables = camelot.read_pdf(pdf_path, pages="1", flavor="stream",flag_size=True)
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
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor,flag_size=True)
            for t in tables:
                df = t.df.replace("\n", " ", regex=True)
                dataframes.append(df)

            if len(dataframes) > 0:
                break

        except Exception:
            continue

    return dataframes





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