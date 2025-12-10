# app.py (complete integrated version)
import os
import io
import base64
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Keep these imports as your project has them
from utils_ocr import (
    convert_pdf_to_images,
    is_vector_pdf,
    extract_tables_with_camelot,
    extract_tables_with_mistral_ocr,
)
from table_postprocess import (
    stitch_multipage_tables,
    normalize_headers_with_subcolumns,
    drop_empty_cols_and_fix_nulls,
    to_csv_download,
    to_excel_download,
)

# ------------------------------
# Small analysis & UI helper functions (integrated here)
# ------------------------------
from io import BytesIO

def display_with_download(df: pd.DataFrame, title: str = "Table"):
    """Display a dataframe in Streamlit and offer CSV/Excel downloads."""
    st.write(f"### {title}")
    st.dataframe(df, use_container_width=True)

    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name=f"{title.replace(' ', '_')}.csv",
        mime="text/csv"
    )

    # Excel download
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=title[:30])
    excel_data = buffer.getvalue()
    st.download_button(
        label="⬇ Download Excel",
        data=excel_data,
        file_name=f"{title.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def analyze_period(r0, r1):
    """
    Given two iterables:
      - r0: period names (e.g., 'Induction', 'Maintenance', ...)
      - r1: visit names (e.g., 'V1', 'V2', ...)
    Return mapping period -> list(visits)
    """
    result = {}
    for period, visit in zip(r0, r1):
        result.setdefault(period, []).append(visit)
    return result


def analyze_row_filled_columns(df: pd.DataFrame, start_row: int = 5, visit_row: int = 1):
    """
    For each data row (starting from start_row), create mapping of row_name -> visits where non-empty value exists.
    visit_row: row index (0-based) where visit labels are present.
    Returns dict: row_name -> {"visits": [...], "count": n}
    """
    result = {}
    # Extract visit names (1..end) from the visit_row row
    # We'll take columns from index 1 onwards as visit columns (col 0 is the row name)
    visit_names = df.iloc[visit_row].tolist()
    visit_names = [str(v).replace("\n", " ").strip() for v in visit_names]

    for row_idx in range(start_row, df.shape[0]):
        row_name = str(df.iloc[row_idx, 0]).strip()
        mapped_visits = []
        for col_idx in range(1, df.shape[1]):
            try:
                val = str(df.iloc[row_idx, col_idx]).strip()
            except Exception:
                val = ""
            if val not in ["", "nan", "None", "none"]:
                # map to visit name if exists, else empty string
                if col_idx < len(visit_names):
                    mapped_visits.append(visit_names[col_idx])
                else:
                    mapped_visits.append(f"col_{col_idx}")
        result[row_name] = {"visits": mapped_visits, "count": len(mapped_visits)}

    return result


def countdots(df: pd.DataFrame, visit_row: int = 1):
    """
    Count non-empty markers (dots/checks/values) under each visit column.
    Assumes visit row index is visit_row and visits start from column 1.
    Returns dict: visit_name -> {"count": n, "rows": [row names]}
    """
    # visits are columns 1..end at visit_row
    visits = df.iloc[visit_row, 1:].tolist()
    visits = [str(v).replace("\n", " ").strip() for v in visits]
    total = len(visits)
    res = {v: {"count": 0, "rows": []} for v in visits}

    for r in range(visit_row + 1, df.shape[0]):
        row_name = str(df.iloc[r, 0]).strip()
        row_values = df.iloc[r, 1:].tolist()
        for j in range(min(len(row_values), total)):
            cell = str(row_values[j]).strip()
            if cell not in ["", " ", "None", "none", "NaN", "nan"]:
                visit_name = visits[j]
                res[visit_name]["count"] += 1
                res[visit_name]["rows"].append(row_name)
    return res

# ------------------------------
# Setup
# ------------------------------
load_dotenv()
# Prefer reading from environment; keep fallback to hard-coded if needed
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "xxxx")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment or .env. Please add it and restart.")
    st.stop()

st.set_page_config(page_title="Phase 2 Clinical Trial Table Extractor", layout="wide")
st.title("Phase 2 Clinical Trial Table Extractor")
st.caption("Upload a PDF, auto-detect and extract multi-page tables with sub-columns using Mistral OCR (Groq).")

# ------------------------------
# Sidebar configuration
# ------------------------------
st.sidebar.header("Extraction settings")
model_name = st.sidebar.text_input("OCR model name", value="mistral-ocr")
dpi = st.sidebar.slider("Image DPI for OCR (scanned PDFs)", min_value=150, max_value=400, value=300, step=25)
max_pages = st.sidebar.number_input("Max pages to process", min_value=1, value=50)
table_hint = st.sidebar.text_area(
    "Optional table hint",
    value="Document Attribute Matrix; if multiple tables exist, focus on the matrix that spans pages."
)
st.sidebar.caption("Hint improves detection for multi-page matrix.")

st.sidebar.header("Post-processing")
merge_similar_headers = st.sidebar.checkbox("Normalize headers & sub-columns", value=True)
clean_nulls = st.sidebar.checkbox("Clean nulls and drop empty columns", value=True)

# ------------------------------
# File upload
# ------------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    # Save to temp for libraries that require a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tmp_pdf_path = tmp_pdf.name

    # Basic PDF info
    try:
        reader = PdfReader(tmp_pdf_path)
        num_pages = len(reader.pages)
    except Exception:
        num_pages = "Unknown"

    st.info(f"PDF loaded: {getattr(uploaded_file, 'name', 'uploaded_file')} • Pages: {num_pages}")

    # Decide strategy (vector vs scanned)
    vector_flag = is_vector_pdf(tmp_pdf_path)

    st.write("Extraction strategy:")
    if vector_flag:
        st.success("Vector-based PDF detected. Attempting Camelot extraction first.")
    else:
        st.warning("Scanned or complex PDF detected. Using Mistral OCR via Groq.")

    # Tabs for output
    tab1, tab2 = st.tabs(["Extracted Table", "Raw Extractions (Debug)"])

      # ------------------------------
    # Extraction
    # ------------------------------
    with st.spinner("Extracting tables..."):
        raw_tables = None
        if vector_flag:
            # First try Camelot
            camelot_tables = extract_tables_with_camelot(tmp_pdf_path, max_pages=max_pages)
            if len(camelot_tables) == 0:
                st.warning("Camelot found no tables or structure is too complex. Falling back to Mistral OCR.")
                images = convert_pdf_to_images(tmp_pdf_path, dpi=dpi, max_pages=max_pages)
                ocr_tables = extract_tables_with_mistral_ocr(
                    images=images,
                    api_key=GROQ_API_KEY,
                    model_name=model_name,
                    table_hint=table_hint,
                )
                raw_tables = ocr_tables
            else:
                raw_tables = camelot_tables
        else:
            # Use OCR path
            images = convert_pdf_to_images(tmp_pdf_path, dpi=dpi, max_pages=max_pages)
            raw_tables = extract_tables_with_mistral_ocr(
                images=images,
                api_key=GROQ_API_KEY,
                model_name=model_name,
                table_hint=table_hint,
            )

    # ------------------------------
    # Debug tab: show raw tables per page
    # ------------------------------
    with tab2:
        st.subheader("Raw per-page extraction")
        if isinstance(raw_tables, dict):
            # OCR returns dict: {page_index: dataframe or list-of-rows}
            for page_idx, page_data in raw_tables.items():
                st.markdown(f"#### Page {page_idx + 1}")
                if isinstance(page_data, pd.DataFrame):
                    st.dataframe(page_data, use_container_width=True)
                else:
                    st.write(page_data)
        else:
            # Camelot returns list of dataframes
            for i, df in enumerate(raw_tables):
                st.markdown(f"#### Table {i + 1}")
                st.dataframe(df, use_container_width=True)

    # ------------------------------
    # Stitch + Clean post-processing
    # ------------------------------
    with st.spinner("Stitching multi-page table and cleaning…"):
        stitched_df = stitch_multipage_tables(raw_tables)

        if merge_similar_headers:
            stitched_df = normalize_headers_with_subcolumns(stitched_df)

        if clean_nulls:
            stitched_df = drop_empty_cols_and_fix_nulls(stitched_df)

    

    # ------------------------------
    # Final display + integrated analyses
    # ------------------------------
    with tab1:
        st.subheader("Extracted Table from PDF")
        if stitched_df is not None and not stitched_df.empty:
            st.dataframe(stitched_df, use_container_width=True)

            # Provide raw CSV / Excel downloads too (original download helpers)
            csv_bytes = to_csv_download(stitched_df)
            xlsx_bytes = to_excel_download(stitched_df)

            st.download_button(
                label="Download extracted table (CSV)",
                data=csv_bytes,
                file_name="extracted_table.csv",
                mime="text/csv",
            )
            st.download_button(
                label="Download extracted table (Excel)",
                data=xlsx_bytes,
                file_name="extracted_table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            from utils_ocr import (
                BLUE_MIN_B,
                BLUE_MAX_R,
                BLUE_MAX_G,
                LINE_TOL,
                TOKEN_GAP_MAX,
                MEANING_OVERRIDES,
                _to_rgb,
                _is_blue,
                _line_key,
                _char_center_bottom_coords,
                sanitize_filename,
            )
            # st.subheader("Blue color thresholds & character grouping")

            # threshold_rows = [
            #     ("BLUE_MIN_B", BLUE_MIN_B, "Min blue channel to consider text BLUE"),
            #     ("BLUE_MAX_R", BLUE_MAX_R, "Max red channel to allow for blue color"),
            #     ("BLUE_MAX_G", BLUE_MAX_G, "Max green channel to allow for blue color"),
            #     ("LINE_TOL", LINE_TOL, "Tolerance for grouping characters into same line (units)"),
            #     ("TOKEN_GAP_MAX", TOKEN_GAP_MAX, "Max gap between tokens (characters/words)"),
            # ]

            # df_thresholds = pd.DataFrame(threshold_rows, columns=["Parameter", "Value", "Note"])
            # st.dataframe(df_thresholds, use_container_width=True)
            # display_with_download(df_thresholds, "Blue Thresholds and Grouping")

            # st.subheader("Meaning overrides (annotation keys -> full descriptions)")
            meaning_df = pd.DataFrame(list(MEANING_OVERRIDES.items()), columns=["Key", "Description"])
            # st.dataframe(meaning_df, use_container_width=True)
            display_with_download(meaning_df, "Meaning Overrides")

            # ------------------------------
            # Analysis: Period Analysis (period row -> visit row)
            # ------------------------------
            try:
                # According to your choice: visit row is row index 1, period row is row index 0
                period_row_index = 0
                visit_row_index = 4

                r0 = stitched_df.iloc[period_row_index, 1:].tolist()
                r1 = stitched_df.iloc[visit_row_index, 1:].tolist()

                # Clean strings
                r0 = [str(x).replace("\n", " ").strip() for x in r0]
                r1 = [str(x).replace("\n", " ").strip() for x in r1]

                period_map = analyze_period(r0, r1)
                period_items = [
                    {"Period": p, "Visit Count": len(v), "Visits": ", ".join(v)}
                    for p, v in period_map.items()
                ]
                period_df = pd.DataFrame(period_items)

                st.subheader("Period Analysis")
                display_with_download(period_df, "Period Analysis")
            except Exception as e:
                st.warning(f"Period analysis failed: {e}")

            # ------------------------------
            # Analysis: Sample (row) -> visits mapping
            # ------------------------------
            try:
                st.subheader("Visits and Counts per Sample")
                # Choose defaults: start scanning samples at row 5 (0-based), can be adjusted later
                mapping = analyze_row_filled_columns(stitched_df, start_row=5, visit_row=visit_row_index)
                formatted_output = [
                    {"Sample": row_name, "Visits": ", ".join(data["visits"]), "Count": data["count"]}
                    for row_name, data in mapping.items()
                ]
                map_df = pd.DataFrame(formatted_output)
                display_with_download(map_df, "Visit Mapping")
            except Exception as e:
                st.warning(f"Sample->Visit analysis failed: {e}")

            # ------------------------------
            # Analysis: Dot count per visit
            # ------------------------------
            try:
                st.subheader(" Visit wise Count")
                dot_res = countdots(stitched_df, visit_row=visit_row_index)
                output = [
                    {"Visit": visit, "Count": info["count"], "Rows": ", ".join(info["rows"])}
                    for visit, info in dot_res.items()
                ]
                dot_df = pd.DataFrame(output)
                display_with_download(dot_df, "Visit Analysis")
            except Exception as e:
                st.warning(f"Dot count analysis failed: {e}")

        else:
            st.error("No final table produced. Try adjusting DPI, hints, or increase max pages.")
else:
    st.info("Upload a PDF to begin.")
