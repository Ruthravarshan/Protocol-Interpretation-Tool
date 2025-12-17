# table_postprocess.py
import io
from io import BytesIO
import pandas as pd
import re
from typing import List, Dict, Any

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


def shift_columns_for_empty_first_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows after the first 4 rows (index >= 4): if multiple consecutive rows have empty first column,
    shift those rows' columns to the left (remove first column's empty cell and shift remaining values left).
    This handles cases where data is misaligned and should be shifted to align with headers.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Identify rows with empty first column starting from row 4 (0-indexed, so index >= 4)
    rows_with_empty_first = []
    for row_idx in range(4, df.shape[0]):
        first_col_val = str(df.iloc[row_idx, 0]).strip()
        
        # Check if first column is empty (None-like)
        if first_col_val in ["", "None", "none", "null", "NULL", "nan", "NaN"]:
            rows_with_empty_first.append(row_idx)
    
    # Find consecutive groups of rows with empty first column
    consecutive_groups = []
    if rows_with_empty_first:
        current_group = [rows_with_empty_first[0]]
        for i in range(1, len(rows_with_empty_first)):
            if rows_with_empty_first[i] == rows_with_empty_first[i-1] + 1:
                # Consecutive row with empty first column
                current_group.append(rows_with_empty_first[i])
            else:
                # Gap found, save group if it has 2+ consecutive rows
                if len(current_group) > 1:
                    consecutive_groups.extend(current_group)
                current_group = [rows_with_empty_first[i]]
        
        # Don't forget the last group
        if len(current_group) > 1:
            consecutive_groups.extend(current_group)
    
    # Shift columns for identified rows with multiple consecutive empty first columns
    for row_idx in consecutive_groups:
        row_values = df.iloc[row_idx].tolist()
        # Shift left: remove first element and append empty value at end to maintain column count
        shifted_values = row_values[1:] + [""]
        df.iloc[row_idx] = shifted_values
    
    return df


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

    # Shift columns for rows with multiple consecutive empty first columns
    # (handles misaligned data where values should be shifted left)
    stitched = shift_columns_for_empty_first_col(stitched)

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
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Extraction", index=False)
    buffer.seek(0)
    return buffer.read()


def split_merged_visit_columns(df):
    """
    Detect and split merged visit columns where consecutive numbers 
    are incorrectly combined (e.g., "12" should be "1" and "2").
    
    Look for patterns in visit rows where numbers appear concatenated
    and split them into separate columns.
    """
    import re
    
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # First, find potential visit rows by looking for numeric patterns
    visit_row_candidates = []
    for idx in range(min(10, len(df_copy))):  # Check first 10 rows
        row_text = ' '.join([str(cell).lower() for cell in df_copy.iloc[idx]])
        if re.search(r'visit|v\d', row_text):
            visit_row_candidates.append(idx)
    
    if not visit_row_candidates:
        return df_copy
    
    # For each visit row candidate, look for merged columns
    for visit_row_idx in visit_row_candidates:
        cols_to_split = []
        
        for col_idx in range(1, len(df_copy.columns)):  # Skip first column
            cell_value = str(df_copy.iloc[visit_row_idx, col_idx]).strip()
            
            # Check if cell contains consecutive digits that should be split
            # Pattern: "12", "123", "1234", etc.
            if re.match(r'^\d{2,4}$', cell_value):
                # Split into individual digits
                individual_digits = [char for char in cell_value]
                cols_to_split.append((col_idx, individual_digits))
        
        # If we found columns to split, modify the dataframe
        if cols_to_split:
            print(f"Found merged columns to split in row {visit_row_idx}: {cols_to_split}")
            
            # Process from right to left to avoid index shifting issues
            for col_idx, digits in reversed(cols_to_split):
                # Insert new columns for each digit except the first
                for i, digit in enumerate(digits):
                    if i == 0:
                        # Replace the original column with the first digit
                        df_copy.iloc[visit_row_idx, col_idx] = digit
                    else:
                        # Insert new column for additional digits
                        new_col_idx = col_idx + i
                        # Create new column
                        df_copy.insert(new_col_idx, f"col_{new_col_idx}", "")
                        # Set the value for this visit row
                        df_copy.iloc[visit_row_idx, new_col_idx] = digit
                        
                        # Copy values from subsequent rows if they exist
                        for row_idx in range(len(df_copy)):
                            if row_idx != visit_row_idx:
                                # Initialize empty for new column
                                df_copy.iloc[row_idx, new_col_idx] = ""
            
            # Only process the first visit row found to avoid complications
            break
    
    return df_copy


# -------------------------
# Superscript extraction and formatting
# -------------------------

_SUP_TAG_PATTERN = re.compile(r"<s>(.*?)</s>", flags=re.IGNORECASE | re.DOTALL)

def extract_superscripts_from_text(text: Any) -> List[str]:
    """
    Return a list of superscript strings found inside <s>...</s> for a single cell.
    If no tags are present or input isn't a string, returns [].
    """
    if not isinstance(text, str):
        return []
    return [m.group(1) for m in _SUP_TAG_PATTERN.finditer(text)]


def collect_superscripts_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan a DataFrame to collect all superscripts.
    Returns a tidy DataFrame with columns:
      - row_idx: int (0-based row index)
      - col_idx: int (0-based column index)
      - column:  column name
      - cell_text: original cell text (with tags)
      - superscript: the text inside <s>...</s>
    Multiple superscripts in the same cell produce multiple rows.
    """
    records: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return pd.DataFrame(columns=["row_idx", "col_idx", "column", "cell_text", "superscript"])

    n_rows, n_cols = df.shape
    for r in range(n_rows):
        for c in range(n_cols):
            cell = df.iat[r, c]
            supers = extract_superscripts_from_text(cell)
            for s in supers:
                records.append({
                    "row_idx": r,
                    "col_idx": c,
                    "column": df.columns[c],
                    "cell_text": cell,
                    "superscript": s,
                })
    return pd.DataFrame.from_records(records)


def collect_superscripts_all(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Collect superscripts across multiple tables.
    Adds a 'table_index' column to indicate which input df the row came from.
    """
    out = []
    for i, df in enumerate(tables):
        tmp = collect_superscripts_df(df)
        if not tmp.empty:
            tmp.insert(0, "table_index", i)
            out.append(tmp)
    if out:
        return pd.concat(out, axis=0, ignore_index=True)
    return pd.DataFrame(columns=["table_index", "row_idx", "col_idx", "column", "cell_text", "superscript"])


def unique_superscripts_list(df_or_df_supers: pd.DataFrame) -> List[str]:
    """
    Given either:
      - the original DataFrame (will extract first), or
      - the collected superscript DataFrame returned by collect_superscripts_df/collect_superscripts_all,
    return a unique, order-preserving list of superscripts found.
    """
    # If it looks like the collected dataframe, read directly
    if "superscript" in df_or_df_supers.columns:
        series = df_or_df_supers["superscript"].astype(str)
    else:
        # Otherwise, treat it like the original table and extract first
        df_supers = collect_superscripts_df(df_or_df_supers)
        series = df_supers["superscript"].astype(str)

    seen = set()
    unique_ordered = []
    for s in series:
        if s not in seen:
            seen.add(s)
            unique_ordered.append(s)
    return unique_ordered


def process_annotations(annotations: List[str]) -> List[str]:
    """
    Process a list of annotations according to the specified rules:
    1. Split on commas.
    2. Trim whitespace.
    3. Keep non-empty and length <= 3.
    4. Sort numerically if all numeric, else lexicographically.
    5. Remove duplicates.
    Returns the processed list.
    """
    # Step 1 & 2: Split on comma and trim
    processed = []
    for ann in annotations:
        parts = [part.strip() for part in str(ann).split(',')]
        processed.extend(parts)
    
    # Step 3: Keep non-empty and length <= 3
    processed = [p for p in processed if p and len(p) <= 3]
    
    # Step 4 & 5: Check if all numeric and sort
    try:
        # Try to convert all to float
        nums = [float(p) for p in processed]
        # If all are numeric, sort numerically and remove duplicates
        processed = sorted(set(processed), key=lambda x: float(x))
    except ValueError:
        # Not all numeric, sort lexicographically and remove duplicates
        processed = sorted(set(processed))
    
    return processed


def to_excel_download_rich_superscripts(df: pd.DataFrame) -> bytes:
    """
    Return XLSX bytes for download with true superscript formatting for segments
    marked by Camelot tags <s>...</s> in cell text.
    """
    if df is None:
        return b""

    try:
        import xlsxwriter
    except ImportError:
        # Fallback to regular export if xlsxwriter not available
        return to_excel_download(df)

    buffer = io.BytesIO()
    workbook = xlsxwriter.Workbook(buffer, {'in_memory': True})
    worksheet = workbook.add_worksheet("Extraction")
    fmt_header = workbook.add_format({'bold': True})
    fmt_sup = workbook.add_format({'font_script': 1})  # 1 = superscript

    pattern = re.compile(r"<s>(.*?)</s>", flags=re.IGNORECASE | re.DOTALL)

    def to_rich_parts(s):
        """
        Convert a string containing <s>...</s> into xlsxwriter rich parts.
        Return list suitable for write_rich_string, or plain string if no tags.
        """
        if not isinstance(s, str) or "<s>" not in s:
            return s

        parts = []
        pos = 0
        for m in pattern.finditer(s):
            start, end = m.span()
            # Preceding plain segment
            if start > pos:
                parts.append(s[pos:start])
            # Superscript segment
            sup_text = m.group(1)
            parts.append(fmt_sup)
            parts.append(sup_text)
            pos = end
        # Trailing plain segment
        if pos < len(s):
            parts.append(s[pos:])
        return parts

    # Write headers
    for c, col in enumerate(df.columns):
        worksheet.write(0, c, col, fmt_header)

    # Write data rows (rich strings if needed)
    n_rows = len(df)
    n_cols = len(df.columns)
    for r in range(n_rows):
        for c in range(n_cols):
            val = df.iat[r, c]
            rich = to_rich_parts(val)
            if isinstance(rich, list):
                try:
                    worksheet.write_rich_string(r + 1, c, *rich)
                except Exception:
                    worksheet.write(r + 1, c, str(val))
            else:
                worksheet.write(r + 1, c, rich)

    workbook.close()
    buffer.seek(0)
    return buffer.read()
