# table_postprocess.py
import io
from io import BytesIO
import pandas as pd

# -------------------------
# Filtering helpers
# -------------------------

def filter_tables_with_many_nones(dfs, threshold=0.3):
    """
    Remove tables where a large portion of cells are None-like.
    `threshold` is fraction of None-like cells above which the table is discarded.
    """
    if not dfs:
        return []

    cleaned = []
    for df in dfs:
        if df is None or df.empty:
            continue
        total = df.size
        if total == 0:
            continue
        none_count = df.astype(str).isin(["None", "none", "NULL", "null", "nan", "NaN"]).sum().sum()
        fraction = none_count / total
        if fraction < threshold:
            cleaned.append(df)
    return cleaned


def remove_rows_with_many_nones(df: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
    """
    Remove rows where a large portion of cells are None-like.
    Keep rows where fraction of None-like cells is less than threshold.
    """
    if df is None or df.empty:
        return df

    rows = []
    for _, row in df.iterrows():
        row_str = row.astype(str)
        total = len(row_str)
        if total == 0:
            continue
        none_count = row_str.isin(["None", "none", "NULL", "null", "nan", "NaN"]).sum()
        fraction = none_count / total
        if fraction < threshold:
            rows.append(row)

    if rows:
        cleaned = pd.DataFrame(rows, columns=df.columns)
        cleaned = cleaned.reset_index(drop=True)
        return cleaned
    else:
        # return empty dataframe with same columns if nothing remains
        return pd.DataFrame(columns=df.columns)


def remove_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are exact duplicates of the column headers (after normalization).
    Also drop fully duplicate rows (post-normalization).
    """
    if df is None or df.empty:
        return df

    # Work on normalized copy so comparison is reliable
    tmp = df.copy().applymap(lambda x: str(x).strip().lower())

    header_values = [str(c).strip().lower() for c in tmp.columns]
    cleaned_rows = []
    for _, row in tmp.iterrows():
        row_values = [str(v).strip().lower() for v in row.tolist()]
        if row_values == header_values:
            continue
        cleaned_rows.append(row)

    if not cleaned_rows:
        return pd.DataFrame(columns=df.columns)

    cleaned_df = pd.DataFrame(cleaned_rows, columns=df.columns)

    # Drop duplicate rows (exact duplicates) preserving first occurrence
    cleaned_df = cleaned_df.drop_duplicates(keep="first").reset_index(drop=True)
    return cleaned_df

# -------------------------
# Header helpers
# -------------------------

def fill_empty_header_with_previous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix ONLY the first header row: any empty header cell inherits the previous non-empty header value.
    After this operation, the first row values are kept (not removed) — caller may drop the header row if desired.
    NOTE: This function replaces the first row values in-place and returns the DataFrame.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    # Ensure there is at least one row to examine
    if df.shape[0] < 1:
        return df

    header = df.iloc[0].astype(str).str.strip().tolist()
    new_header = []
    last_value = ""
    for val in header:
        val_str = str(val).strip()
        if val_str == "" or val_str.lower() in ["none", "nan", "null"]:
            new_header.append(last_value)
        else:
            new_header.append(val_str)
            last_value = val_str

    # Replace the FIRST ROW with cleaned version (we keep it as a data row here;
    # higher-level stitcher may set df.columns from it or drop it)
    df.iloc[0] = new_header
    return df


def build_header_from_multiple_rows(df: pd.DataFrame, max_header_rows: int = 5) -> pd.DataFrame:
    """
    Build a single header row from multiple top rows (0..N) by concatenating
    non-empty values vertically per column. Then drop the header-rows from the dataframe.
    This is useful when header spans multiple lines.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    nrows = min(max_header_rows, df.shape[0])

    header_rows = []
    # collect header rows until we hit a row that looks like data (heuristic: has no blanks)
    for r in range(nrows):
        row = df.iloc[r].astype(str).str.strip().tolist()
        header_rows.append(row)
        # stop early if this row appears fully filled (no blanks)
        if all([v != "" and v.lower() not in ["none", "nan", "null"] for v in row]):
            break

    # Merge header rows into final header
    final_header = []
    ncols = df.shape[1]
    for c in range(ncols):
        parts = []
        for row in header_rows:
            v = row[c]
            if v != "" and v.lower() not in ["none", "nan", "null"]:
                parts.append(v)
        if parts:
            final_header.append(" ".join(parts))
        else:
            final_header.append(f"col_{c}")

    # assign new header and drop header rows from top
    df.columns = final_header
    df = df.iloc[len(header_rows):].reset_index(drop=True)
    return df


# -------------------------
# Normalization + cleanup
# -------------------------

def normalize_headers_with_subcolumns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize duplicated headers, trim whitespace, and merge duplicate columns by
    preferring non-empty cells in earlier columns.
    """
    if df is None or df.empty:
        return df

    # Ensure headers are strings
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    seen = {}
    new_cols = []
    for c in df.columns:
        key = c.lower()
        if key in seen:
            prev_idx = seen[key]
            # Merge current column into the first occurrence by keeping non-empty values
            merged = df.iloc[:, prev_idx].astype(str)
            curr = df[c].astype(str)
            # merged.where(condition, other) keeps merged where condition True, else other
            df.iloc[:, prev_idx] = merged.where(merged.str.strip() != "", curr)
        else:
            seen[key] = len(new_cols)
            new_cols.append(c)

    df = df.loc[:, new_cols]
    return df


def drop_empty_cols_and_fix_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace obvious null strings and drop columns that are entirely empty/whitespace.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df = df.replace({"None": "", "none": "", "null": "", "NULL": "", "NaN": "", "nan": ""})

    # Drop columns where all values are blank after replacement
    mask = ~df.apply(lambda col: col.astype(str).str.strip().eq("").all(), axis=0)
    df = df.loc[:, mask]
    return df

# -------------------------
# Header helpers
# -------------------------

def fill_empty_header_with_previous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix ONLY the first header row: any empty header cell inherits the previous non-empty header value.
    After this operation, the first row values are kept (not removed) — caller may drop the header row if desired.
    NOTE: This function replaces the first row values in-place and returns the DataFrame.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    # Ensure there is at least one row to examine
    if df.shape[0] < 1:
        return df

    header = df.iloc[0].astype(str).str.strip().tolist()
    new_header = []
    last_value = ""
    for val in header:
        val_str = str(val).strip()
        if val_str == "" or val_str.lower() in ["none", "nan", "null"]:
            new_header.append(last_value)
        else:
            new_header.append(val_str)
            last_value = val_str

    # Replace the FIRST ROW with cleaned version (we keep it as a data row here;
    # higher-level stitcher may set df.columns from it or drop it)
    df.iloc[0] = new_header
    return df


def build_header_from_multiple_rows(df: pd.DataFrame, max_header_rows: int = 5) -> pd.DataFrame:
    """
    Build a single header row from multiple top rows (0..N) by concatenating
    non-empty values vertically per column. Then drop the header-rows from the dataframe.
    This is useful when header spans multiple lines.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    nrows = min(max_header_rows, df.shape[0])

    header_rows = []
    # collect header rows until we hit a row that looks like data (heuristic: has no blanks)
    for r in range(nrows):
        row = df.iloc[r].astype(str).str.strip().tolist()
        header_rows.append(row)
        # stop early if this row appears fully filled (no blanks)
        if all([v != "" and v.lower() not in ["none", "nan", "null"] for v in row]):
            break

    # Merge header rows into final header
    final_header = []
    ncols = df.shape[1]
    for c in range(ncols):
        parts = []
        for row in header_rows:
            v = row[c]
            if v != "" and v.lower() not in ["none", "nan", "null"]:
                parts.append(v)
        if parts:
            final_header.append(" ".join(parts))
        else:
            final_header.append(f"col_{c}")

    # assign new header and drop header rows from top
    df.columns = final_header
    df = df.iloc[len(header_rows):].reset_index(drop=True)
    return df


# -------------------------
# Normalization + cleanup
# -------------------------

def normalize_headers_with_subcolumns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize duplicated headers, trim whitespace, and merge duplicate columns by
    preferring non-empty cells in earlier columns.
    """
    if df is None or df.empty:
        return df

    # Ensure headers are strings
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    seen = {}
    new_cols = []
    for c in df.columns:
        key = c.lower()
        if key in seen:
            prev_idx = seen[key]
            # Merge current column into the first occurrence by keeping non-empty values
            merged = df.iloc[:, prev_idx].astype(str)
            curr = df[c].astype(str)
            # merged.where(condition, other) keeps merged where condition True, else other
            df.iloc[:, prev_idx] = merged.where(merged.str.strip() != "", curr)
        else:
            seen[key] = len(new_cols)
            new_cols.append(c)

    df = df.loc[:, new_cols]
    return df


def drop_empty_cols_and_fix_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace obvious null strings and drop columns that are entirely empty/whitespace.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df = df.replace({"None": "", "none": "", "null": "", "NULL": "", "NaN": "", "nan": ""})

    # Drop columns where all values are blank after replacement
    mask = ~df.apply(lambda col: col.astype(str).str.strip().eq("").all(), axis=0)
    df = df.loc[:, mask]
    return df

# -------------------------
# Stitching pipeline
# -------------------------

def stitch_multipage_tables(raw) -> pd.DataFrame:
    """
    Stitch a collection of tables (either dict page->DataFrame or list of DataFrames)
    into a single DataFrame with aligned columns. Performs filtering and cleaning:
      - removes empty tables
      - optionally filters out tables full of None-like values
      - aligns columns (union of columns)
      - concatenates
      - row-level removal of None-like rows
      - removal of repeated header rows
      - fixes header blanks in first row
      - normalize headers and drop empty columns
    """
    if raw is None:
        return pd.DataFrame()

    # collect dataframes in order
    if isinstance(raw, dict):
        dfs = [raw[k] for k in sorted(raw.keys()) if isinstance(raw[k], pd.DataFrame)]
    else:
        dfs = [df for df in raw if isinstance(df, pd.DataFrame)]

    # drop empties
    dfs = [df for df in dfs if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()

    # optional: remove entirely useless tables (uncomment if desired)
    # dfs = filter_tables_with_many_nones(dfs, threshold=0.3)

    # Build master column list preserving order from first df
    master_cols = list(dfs[0].columns)
    for df in dfs[1:]:
        for c in df.columns:
            if c not in master_cols:
                master_cols.append(c)

    # Reindex each df to master columns (fills with NaN where missing)
    aligned = []
    for df in dfs:
        aligned_df = df.reindex(columns=master_cols)
        aligned.append(aligned_df)

    # Concatenate
    stitched = pd.concat(aligned, axis=0, ignore_index=True)

    # Row-level cleaning: remove rows mostly composed of None-like values
    stitched = remove_rows_with_many_nones(stitched, threshold=0.5)

    # Remove repeated header rows that appear on every page
    stitched = remove_repeated_header_rows(stitched)

    # Fix blanks in the first row (fill from previous column)
    # This function keeps first-row as data row; if your pipeline expects df.columns to be header,
    # you may later choose to set df.columns = df.iloc[0] and drop row 0. For now we only fix blanks.
    stitched = fill_empty_header_with_previous(stitched)

    # Normalize duplicate headers and merge columns where headers are the same
    stitched = normalize_headers_with_subcolumns(stitched)

    # Replace null-like strings and drop empty columns
    stitched = drop_empty_cols_and_fix_nulls(stitched)

    # Reset index and return
    stitched = stitched.reset_index(drop=True)
    return stitched


# -------------------------
# Download helpers
# -------------------------
def to_csv_download(df: pd.DataFrame) -> bytes:
    """
    Return CSV bytes for download. Uses UTF-8 encoding.
    """
    if df is None:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def to_excel_download(df: pd.DataFrame) -> bytes:
    """
    Return XLSX bytes for download.
    """
    if df is None:
        return b""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Extraction", index=False)
    buffer.seek(0)
    return buffer.read()

