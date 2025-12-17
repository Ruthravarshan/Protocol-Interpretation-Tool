# test.py - Integrated Clinical Trial Table Extractor with Advanced RAG Analytics
import os
import tempfile
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
import re
import json
import numpy as np
from typing import List, Dict, Any

# Import from existing modules
from utils_ocr import (
    extract_tables_with_camelot,
    MEANING_OVERRIDES,
)
from table_postproccess import (
    stitch_multipage_tables,
    normalize_headers_with_subcolumns,
    drop_empty_cols_and_fix_nulls,
    to_csv_download,
    to_excel_download_rich_superscripts,
    unique_superscripts_list,
    process_annotations,
)
from annotation_extractor import (
    extract_annotations_from_dataframe,
)

# RAG system imports
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.generativeai as genai
import pdfplumber

# Try to import Groq for fallback
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Try FAISS; fall back to NumPy if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

try:
    from google.api_core.exceptions import ResourceExhausted as GoogleResourceExhausted
except Exception:
    GoogleResourceExhausted = None

###############################################################################
# Configuration
###############################################################################
EMBED_MODEL = "all-MiniLM-L6-v2"
TEMPERATURE = 0.2
MAX_TABLE_ROWS_TO_INDEX = 500
MAX_PDF_PAGES_TO_INDEX = None
RETRIEVE_TOP_K = 8
CONTEXT_SNIPPET_LIMIT = 15
CHUNK_MAX_LEN = 600
CHUNK_OVERLAP = 50

GEMINI_PREFERRED = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-pro",
]

###############################################################################
# Table Analysis Functions from app.py
###############################################################################

def find_period_and_visit_rows(df: pd.DataFrame):
    """
    Dynamically find period and visit rows by searching for 'period' and 'visit' keywords.
    Enhanced rule implementation for visit row finding.
    """
    period_row_index = None
    visit_row_index = None
    
    # Search for period row first
    for row_idx in range(df.shape[0]):
        row_text = ' '.join([str(df.iloc[row_idx, col]).lower() for col in range(df.shape[1])])
        if 'period' in row_text:
            period_row_index = row_idx
            break
    
    # Search for visit row using enhanced rule
    start_search_idx = period_row_index if period_row_index is not None else 0
    
    # First Rule: Look for rows where first column starts with 'visit', 'visits', or 'v'
    for row_idx in range(start_search_idx, df.shape[0]):
        first_col_text = str(df.iloc[row_idx, 0]).lower().strip()
        
        if (first_col_text.startswith('visit') or 
            first_col_text.startswith('visits') or 
            re.match(r'^v\d*', first_col_text)):
            visit_row_index = row_idx
            break
    
    # Second Rule: If not found, look for visit patterns in other columns
    if visit_row_index is None:
        for row_idx in range(start_search_idx, df.shape[0]):
            other_cols_text = ' '.join([str(df.iloc[row_idx, col]).lower() for col in range(1, df.shape[1])])
            
            if re.search(r"visits?|v\d", other_cols_text):
                current_visit_count = len(re.findall(r"visits?|v\d", other_cols_text))
                
                if current_visit_count > 1:
                    visit_row_index = row_idx
                    break
                elif row_idx + 1 < df.shape[0]:
                    next_row_other_cols = ' '.join([str(df.iloc[row_idx + 1, col]).lower() for col in range(1, df.shape[1])])
                    visit_pattern_count = len(re.findall(r"visits?|v\d", next_row_other_cols))
                    if visit_pattern_count > 1:
                        visit_row_index = row_idx
                        break
                    else:
                        visit_row_index = row_idx
                        break
                else:
                    visit_row_index = row_idx
                    break
    
    return period_row_index, visit_row_index

def analyze_period(r0, r1):
    """Given period names and visit names, return mapping period -> list(visits)"""
    result = {}
    for period, visit in zip(r0, r1):
        result.setdefault(period, []).append(visit)
    return result

def find_first_data_row_for_mapping(df, visit_row_index):
    """Find the first row after visit_row_index that has minimum 1 and maximum 2 columns filled with symbols."""
    target_symbols = {".", "x", "●", "○", "X", "•"}
    
    for row_idx in range(visit_row_index + 1, df.shape[0]):
        filled_count = 0
        for col_idx in range(1, df.shape[1]):
            try:
                cell = str(df.iloc[row_idx, col_idx]).strip()
                if any(symbol in cell for symbol in target_symbols):
                    filled_count += 1
            except Exception:
                continue
        
        if 1 <= filled_count <= 2:
            return row_idx
    
    return visit_row_index + 1

def analyze_row_filled_columns(df: pd.DataFrame, start_row: int = 5, visit_row: int = 1):
    """For each data row, create mapping of row_name -> visits where non-empty value exists."""
    result = {}
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
                if col_idx < len(visit_names):
                    mapped_visits.append(visit_names[col_idx])
                else:
                    mapped_visits.append(f"col_{col_idx}")
        result[row_name] = {"visits": mapped_visits, "count": len(mapped_visits)}

    return result

def countdots(df: pd.DataFrame, visit_row: int = 1):
    """Count specific symbols under each visit column."""
    target_symbols = {".", "x", "●", "○", "X","•"}
    
    visits = df.iloc[visit_row, 1:].tolist()
    visits = [str(v).replace("\n", " ").strip() for v in visits]
    total = len(visits)
    res = {v: {"count": 0, "rows": []} for v in visits}

    for r in range(visit_row + 1, df.shape[0]):
        row_name = str(df.iloc[r, 0]).strip()
        row_values = df.iloc[r, 1:].tolist()
        for j in range(min(len(row_values), total)):
            cell = str(row_values[j]).strip()
            if any(symbol in cell for symbol in target_symbols):
                visit_name = visits[j]
                res[visit_name]["count"] += 1
                res[visit_name]["rows"].append(row_name)
    return res

def normalize_visit(v):
    """Normalize visit names to standard format"""
    v_lower = v.lower()
    if ("visit" in v_lower) or (re.match(r"v\d+", v_lower)):
        return v
    return f"Visit {v}"

###############################################################################
# RAG System Components
###############################################################################

def simple_clean(text: str) -> str:
    """Clean text for indexing"""
    if not text:
        return ""
    text = text.replace("x00", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def chunk_text(text: str, max_len: int = CHUNK_MAX_LEN, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    i = 0
    step = max(1, max_len - overlap)
    while i < len(words):
        chunk_words = words[i:i + max_len]
        chunks.append(" ".join(chunk_words))
        i += step
    return chunks

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)

class TextIndex:
    """Hybrid dense + sparse text retrieval index"""
    def __init__(self, model_name: str = EMBED_MODEL):
        self.embedder = get_embedder(model_name)
        self.docs: List[Dict[str, Any]] = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        for d in docs:
            d["text"] = simple_clean(d.get("text", ""))
        self.docs.extend(docs)

    def build(self) -> None:
        texts = [d["text"] for d in self.docs]
        if not texts:
            return
        
        print(f"Building embeddings for {len(texts)} text chunks...")
        X = self.embedder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings = np.array(X, dtype=np.float32)

        if FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.faiss_index.add(self.embeddings)

        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = RETRIEVE_TOP_K, alpha_dense: float = 0.65, alpha_sparse: float = 0.35) -> List[Dict[str, Any]]:
        if self.embeddings is None or self.bm25 is None:
            return []

        # Dense search
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        q = np.array(q_emb, dtype=np.float32)

        if FAISS_AVAILABLE and self.faiss_index is not None:
            D, I = self.faiss_index.search(q, k * 3)
            dense_scores = {int(idx): float(score) for idx, score in zip(I[0], D[0]) if idx != -1}
        else:
            sims = (self.embeddings @ q.T).ravel()
            top_idx = np.argsort(-sims)[:k * 3]
            dense_scores = {int(i): float(sims[i]) for i in top_idx}

        # Sparse search
        sparse_raw = self.bm25.get_scores(query.lower().split())
        sparse_scores = {}
        if len(sparse_raw) > 0:
            arr = np.array(sparse_raw, dtype=np.float32)
            maxv = np.max(arr)
            if maxv > 0:
                arr = arr / (maxv + 1e-9)
            for i, sc in enumerate(arr):
                sparse_scores[i] = float(sc)

        # Combine scores
        combined = []
        candidate_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        for i in candidate_ids:
            score = alpha_dense * dense_scores.get(i, 0.0) + alpha_sparse * sparse_scores.get(i, 0.0)
            combined.append((i, score))
        combined.sort(key=lambda x: x[1], reverse=True)

        hits = []
        for i, sc in combined[:k]:
            d = self.docs[i].copy()
            d["score"] = float(sc)
            hits.append(d)
        return hits

def extract_pdf_text_chunks(pdf_path: str, max_pages: int = MAX_PDF_PAGES_TO_INDEX) -> List[Dict[str, Any]]:
    """Extract text chunks from PDF"""
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = list(pdf.pages)
        pages_to_process = pages if max_pages is None else pages[:max_pages]
        for p_idx, page in enumerate(pages_to_process, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = simple_clean(text)
            if not text:
                continue
            chunks = chunk_text(text, max_len=CHUNK_MAX_LEN, overlap=CHUNK_OVERLAP)
            for ci, chunk in enumerate(chunks):
                docs.append({
                    "id": f"pdf:{os.path.basename(pdf_path)}:p{p_idx}:c{ci}",
                    "text": chunk,
                    "metadata": {"source": "pdf", "file": os.path.basename(pdf_path), "page": p_idx, "chunk_idx": ci}
                })
    return docs

def dataframe_to_docs(df: pd.DataFrame, table_name: str = "input", max_rows: int = MAX_TABLE_ROWS_TO_INDEX) -> List[Dict[str, Any]]:
    """Convert DataFrame rows to searchable documents"""
    docs = []
    df = df.copy()
    if "_row_id" not in df.columns:
        df["_row_id"] = [f"r_{i}" for i in range(len(df))]

    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    for _, row in df.iterrows():
        kv_pairs = []
        for c in df.columns:
            val = row[c]
            if pd.isna(val):
                continue
            kv_pairs.append(f"{c}={val}")
        text = "; ".join(kv_pairs)
        rid = row["_row_id"]
        docs.append({
            "id": f"table:{table_name}:{rid}",
            "text": text,
            "metadata": {"source": "table", "table": table_name, "row_id": str(rid)}
        })
    return docs

def table_analysis_to_docs(df: pd.DataFrame, table_name: str = "extracted_table") -> List[Dict[str, Any]]:
    """Convert table analysis results to searchable documents"""
    docs = []
    
    try:
        # Period analysis
        period_row_index, visit_row_index = find_period_and_visit_rows(df)
        if period_row_index is not None and visit_row_index is not None:
            r0 = df.iloc[period_row_index, 1:].tolist()
            r1 = df.iloc[visit_row_index, 1:].tolist()
            r0 = [str(x).replace("\n", " ").strip() for x in r0]
            r1 = [str(x).replace("\n", " ").strip() for x in r1]
            period_map = analyze_period(r0, r1)
            
            period_text = "Period Analysis: "
            for period, visits in period_map.items():
                normalized_visits = [normalize_visit(v) for v in visits]
                period_text += f"{period} includes {len(visits)} visits: {', '.join(normalized_visits)}; "
            
            docs.append({
                "id": f"analysis:{table_name}:period",
                "text": period_text,
                "metadata": {"source": "analysis", "type": "period", "table": table_name}
            })
        
        # Visit mapping analysis
        if visit_row_index is not None:
            dynamic_start_row = find_first_data_row_for_mapping(df, visit_row_index)
            mapping = analyze_row_filled_columns(df, start_row=dynamic_start_row, visit_row=visit_row_index)
            
            mapping_text = "Visit Mapping Analysis: "
            for row_name, data in mapping.items():
                normalized_visits = [normalize_visit(v) for v in data["visits"]]
                mapping_text += f"{row_name} has {data['count']} visits: {', '.join(normalized_visits)}; "
            
            docs.append({
                "id": f"analysis:{table_name}:mapping",
                "text": mapping_text,
                "metadata": {"source": "analysis", "type": "mapping", "table": table_name}
            })
        
        # Visit count analysis
        if visit_row_index is not None:
            dot_res = countdots(df, visit_row=visit_row_index)
            
            count_text = "Visit Count Analysis: "
            for visit, info in dot_res.items():
                count_text += f"{visit} has {info['count']} collections from rows: {', '.join(info['rows'])}; "
            
            docs.append({
                "id": f"analysis:{table_name}:count",
                "text": count_text,
                "metadata": {"source": "analysis", "type": "count", "table": table_name}
            })
        
        # Annotations analysis
        superscripts_list = unique_superscripts_list(df)
        processed_list = process_annotations(superscripts_list)
        if processed_list:
            annotation_text = "Annotations found in table: "
            for sup in processed_list:
                if MEANING_OVERRIDES and sup in MEANING_OVERRIDES:
                    meaning = MEANING_OVERRIDES.get(sup)
                    annotation_text += f"{sup} means {meaning}; "
                else:
                    annotation_text += f"{sup}; "
            
            docs.append({
                "id": f"analysis:{table_name}:annotations",
                "text": annotation_text,
                "metadata": {"source": "analysis", "type": "annotations", "table": table_name}
            })
    
    except Exception as e:
        print(f"Warning: Table analysis failed: {e}")
    
    return docs

###############################################################################
# Gemini Integration
###############################################################################

def _filter_supported_models(models) -> List[str]:
    """Filter supported Gemini models"""
    supported = []
    for m in models:
        methods = getattr(m, "supported_generation_methods", []) or []
        name = m.name
        if ("generateContent" in methods or "generate_content" in methods) and ("exp" not in name):
            supported.append(name)
    return supported

@st.cache_resource(show_spinner=False)
def resolve_and_init_gemini(preferred_names=GEMINI_PREFERRED, temperature: float = TEMPERATURE):
    """Initialize Gemini model"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    genai.configure(api_key=api_key)

    models = list(genai.list_models())
    supported = _filter_supported_models(models)

    chosen = None
    for cand in preferred_names:
        cand_with_prefix = cand if cand.startswith("models/") else f"models/{cand}"
        if cand_with_prefix in supported:
            chosen = cand_with_prefix
            break
    if not chosen:
        if supported:
            chosen = supported[0]
        else:
            raise RuntimeError("No Gemini models supporting generateContent were found.")

    model = genai.GenerativeModel(chosen)
    return model, temperature, chosen

def is_resource_exhausted(exc: Exception) -> bool:
    """Check if exception is due to resource exhaustion"""
    if GoogleResourceExhausted and isinstance(exc, GoogleResourceExhausted):
        return True
    msg = str(exc).lower()
    return "resourceexhausted" in msg or "quota" in msg or "rate limit" in msg

def groq_generate(prompt: str, temperature: float = 0.2) -> str:
    """Generate text using Groq API as fallback"""
    if not GROQ_AVAILABLE:
        raise Exception("Groq library not available")
    
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise Exception("GROQ_API_KEY not found in environment")
    
    client = Groq(api_key=groq_key)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt[:8000]}],
            temperature=temperature,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Groq generation failed: {e}")

def gemini_generate(model, temperature: float, prompt: str) -> str:
    """Generate text using Gemini with Groq fallback"""
    try:
        resp = model.generate_content(prompt, generation_config={"temperature": temperature})
        return (resp.text or "").strip()
    except Exception as e:
        if is_resource_exhausted(e):
            print("🔄 Gemini quota exceeded, switching to Groq...")
            try:
                return groq_generate(prompt, temperature)
            except Exception as groq_error:
                print(f"❌ Groq fallback failed: {groq_error}")
                raise Exception(f"All generation methods failed. Gemini: {e}, Groq: {groq_error}")
        raise e

###############################################################################
# Enhanced RAG Engine
###############################################################################

class EnhancedRAGEngine:
    """Enhanced RAG engine with table analysis integration"""
    def __init__(self, model, temperature: float, embed_model: str = EMBED_MODEL):
        self.index = TextIndex(model_name=embed_model)
        self.model = model
        self.temperature = temperature
        self.table_name = "extracted_table"
        self.retrieve_k = RETRIEVE_TOP_K
        self.context_limit = CONTEXT_SNIPPET_LIMIT

    def ingest_with_analysis(self, df: pd.DataFrame, pdf_path: str, table_name: str = "extracted_table") -> None:
        """Ingest DataFrame and PDF with advanced table analysis"""
        self.table_name = table_name
        
        # Standard document ingestion
        df_docs = dataframe_to_docs(df, table_name=table_name, max_rows=MAX_TABLE_ROWS_TO_INDEX)
        pdf_docs = extract_pdf_text_chunks(pdf_path, max_pages=MAX_PDF_PAGES_TO_INDEX)
        
        # Enhanced: Add table analysis documents
        analysis_docs = table_analysis_to_docs(df, table_name=table_name)
        
        all_docs = df_docs + pdf_docs + analysis_docs
        self.index.add_documents(all_docs)
        self.index.build()
        
        print(f"Indexed {len(df_docs)} table rows, {len(pdf_docs)} PDF chunks, {len(analysis_docs)} analysis results")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        return self.index.search(query, k=self.retrieve_k)

    def build_enhanced_prompt(self, user_prompt: str, retrieved: List[Dict[str, Any]]) -> str:
        """Build enhanced prompt with table analysis rules"""
        instructions = """
You are analyzing a clinical trial protocol and extracted table data to provide comprehensive study logistics analysis.

EXTRACTION RULES APPLIED:
1. Period Analysis: Periods are mapped to visits based on column alignment in extracted tables
2. Visit Row Detection: Enhanced rules to find visit rows by checking first column for 'visit'/'visits'/'v' patterns, then other columns
3. Visit Mapping: Analyzes which samples/rows have data in which visit columns using symbol detection (., x, ●, ○, X, •)
4. Count Analysis: Counts occurrences of collection symbols per visit across all rows
5. Annotation Extraction: Identifies superscripts and their meanings from tables

FOCUS AREAS:
- Number of patients/subjects (sample size, enrollment target, N=)
- Number of trial sites (study sites, clinical sites, centers)
- Visit schedule analysis based on extracted table structure
- Collection patterns from visit mapping analysis
- Period-to-visit relationships from table structure
- Annotation meanings for protocol understanding

CALCULATION METHODS:
- Use extracted visit counts and mappings for accurate totals
- Apply period analysis results for timeline understanding
- Incorporate annotation meanings for protocol interpretation
- Cross-reference PDF content with table analysis results

Rules:
- Output must be formatted as Markdown tables
- Do NOT include citations or source references
- Use actual extracted data patterns and counts
- If values are ranges, show as ranges (e.g., 25-30)
- Include confidence notes for inferred values
- Prioritize table analysis results over general text
"""

        context_lines = []
        for h in retrieved[:self.context_limit]:
            snippet = h["text"]
            source_info = h.get("metadata", {})
            source_type = source_info.get("source", "unknown")
            context_lines.append(f"[{source_type.upper()}] {snippet}")
        
        context = "\n".join(context_lines)

        output_schema = """
Return only these sections in Markdown:

## Summary
Brief overview mentioning patients, sites, study duration, and key findings from table analysis.

## Protocol Information
| Metric | Value | Source |
|---|---|---|
| Protocol Number |  |  |
| Version |  |  |
| Date |  |  |

## Enrollment and Sites
| Metric | Value | Unit | Confidence |
|---|---|---|---|
| Number of patients |  |  |  |
| Number of sites |  |  |  |

## Period Analysis (Based on extracted table rules)
| Period | Visit Count | Visits in which collected |
|---|---|---|
|  |  |  |

## Visit Mapping Analysis (Based on symbol detection rules)
| Sample name | Visits in which collected | Count |
|---|---|---|
|  |  |  |

## Visit Count Analysis (Based on collection symbol rules)
| Visit name | Count | Samples collected |
|---|---|---|
|  |  |  |

## Extracted Annotations (Based on superscript extraction rules)
List the annotations found using the superscript detection rules from the table:
- **Annotation**: Meaning (if available from MEANING_OVERRIDES)

## Data Quality Assessment
- **Table extraction confidence**: [High/Medium/Low]
- **Visit mapping completeness**: [percentage]
- **Missing data areas**: [list]
- **Validation notes**: [observations]

## Notes
- List assumptions made during analysis
- Highlight any discrepancies between PDF and table data
- Note limitations of automated extraction
"""

        final_prompt = f"""
{instructions}

User request: {user_prompt}

Retrieved context and analysis:
{context}

{output_schema}

Generate the analysis using the extracted table data patterns and PDF content.
"""
        return final_prompt

    def generate_enhanced_report(self, user_prompt: str) -> str:
        """Generate enhanced report with table analysis"""
        try:
            # Enhanced search combining multiple aspects
            enhanced_query = f"{user_prompt} patients subjects sites enrollment visit period analysis collection mapping annotations symbols"
            retrieved = self.retrieve(enhanced_query)
            
            prompt = self.build_enhanced_prompt(user_prompt, retrieved)
            return gemini_generate(self.model, self.temperature, prompt)
        except Exception as e:
            if is_resource_exhausted(e):
                self.retrieve_k = max(6, self.retrieve_k // 2)
                self.context_limit = max(10, self.context_limit // 2)
                enhanced_query = f"{user_prompt} patients sites visits"
                retrieved = self.retrieve(enhanced_query)
                prompt = self.build_enhanced_prompt(user_prompt, retrieved)
                return gemini_generate(self.model, self.temperature, prompt)
            raise e

###############################################################################
# Streamlit UI with Enhanced Design
###############################################################################

st.set_page_config(
    page_title="Clinical Trial Analytics Platform", 
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        color: white !important;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .analysis-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧬 Clinical Trial Analytics Platform</h1>
    <p>Advanced PDF extraction, rule-based analysis, and AI-powered insights for clinical trial protocols</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
st.sidebar.markdown("## ⚙️ Configuration")

with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📄 Extraction Settings")
    max_pages = st.number_input("Max pages to process", min_value=1, value=50, help="Limit PDF processing for faster results")
    merge_similar_headers = st.checkbox("🔄 Normalize headers & sub-columns", value=True, help="Clean and standardize table headers")
    clean_nulls = st.checkbox("🧹 Clean nulls and empty columns", value=True, help="Remove empty data for cleaner analysis")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🤖 RAG Settings")
    default_prompt = st.text_area(
        "Analysis Prompt", 
        value="Generate comprehensive report on patient enrollment, sites, visit schedule, and collection patterns based on extracted table analysis and PDF content.",
        height=120,
        help="Customize the AI analysis prompt"
    )
    
    # Model selection
    temperature = st.slider("🌡️ Analysis Creativity", 0.0, 1.0, 0.2, 0.1, help="Higher values = more creative analysis")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick stats if data is available
    if "extracted_df" in st.session_state:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Quick Stats")
        df = st.session_state["extracted_df"]
        st.metric("Table Rows", f"{df.shape[0]:,}")
        st.metric("Table Columns", f"{df.shape[1]:,}")
        if "enhanced_report" in st.session_state:
            st.success("✅ RAG Report Ready")
        st.markdown('</div>', unsafe_allow_html=True)

# File upload with enhanced styling
st.markdown("## 📤 Upload Your Clinical Trial PDF")
st.markdown('<div class="upload-area">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a PDF file", 
    type=["pdf"],
    help="Upload your clinical trial protocol PDF for analysis"
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h3>🎯 Ready for Analysis</h3>
        <p>Upload a clinical trial PDF to extract tables, analyze visit schedules, and generate comprehensive reports</p>
        <br>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div class="feature-card" style="min-width: 200px; max-width: 250px;">
                <h4>🔍 Smart Extraction</h4>
                <p>Advanced table detection with Camelot OCR technology</p>
            </div>
            <div class="feature-card" style="min-width: 200px; max-width: 250px;">
                <h4>📊 Rule-Based Analysis</h4>
                <p>Period mapping, visit analysis, and annotation extraction</p>
            </div>
            <div class="feature-card" style="min-width: 200px; max-width: 250px;">
                <h4>🤖 AI Insights</h4>
                <p>RAG-powered analysis combining extracted data with PDF content</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tmp_pdf_path = tmp_pdf.name

    # Basic PDF info
    try:
        reader = PdfReader(tmp_pdf_path)
        num_pages = len(reader.pages)
    except Exception:
        num_pages = "Unknown"

    # PDF info display with enhanced styling
    st.markdown(f"""
    <div class="success-card">
        <h3>📄 PDF Successfully Loaded</h3>
        <p><strong>File:</strong> {getattr(uploaded_file, 'name', 'uploaded_file')}</p>
        <p><strong>Pages:</strong> {num_pages}</p>
        <p><strong>Status:</strong> Ready for processing</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs with enhanced styling
    tab1, tab2, tab3 = st.tabs(["🔍 Table Extraction & Analysis", "🧠 RAG Analysis", "🐛 Debug Info"])

    # Tab 1: Table Extraction
    with tab1:
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2>🔍 Table Extraction & Rule-Based Analysis</h2>
            <p>Advanced PDF table detection and clinical trial schedule analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Extraction progress
        progress_container = st.container()
        
        with st.spinner("🔄 Extracting tables with Camelot OCR..."):
            raw_tables = extract_tables_with_camelot(tmp_pdf_path, max_pages=max_pages)

        if len(raw_tables) == 0:
            st.markdown("""
            <div style="background: #ff6b6b; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h3>⚠️ No Tables Detected</h3>
                <p>The PDF doesn't contain extractable tables or they may be in image format</p>
                <p><small>Try adjusting extraction settings or ensure tables are text-based</small></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Process tables
            with st.spinner("🛠️ Processing and cleaning extracted tables..."):
                stitched_df = stitch_multipage_tables(raw_tables)
                
                if merge_similar_headers:
                    stitched_df = normalize_headers_with_subcolumns(stitched_df)
                
                if clean_nulls:
                    stitched_df = drop_empty_cols_and_fix_nulls(stitched_df)

            if stitched_df is not None and not stitched_df.empty:
                # Success metrics
                st.markdown(f"""
                <div class="metric-card">
                    <h3>✅ Extraction Complete!</h3>
                    <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                        <div><strong>{stitched_df.shape[0]:,}</strong><br>Rows</div>
                        <div><strong>{stitched_df.shape[1]:,}</strong><br>Columns</div>
                        <div><strong>{len(raw_tables)}</strong><br>Tables Found</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Table display
                st.markdown("### 📊 Extracted Table Data")
                st.dataframe(stitched_df, use_container_width=True, height=400)
                
                # Store in session state for RAG
                st.session_state["extracted_df"] = stitched_df
                st.session_state["pdf_path"] = tmp_pdf_path

                # Download options with enhanced styling
                st.markdown("### 💾 Download Options")
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv_bytes = to_csv_download(stitched_df)
                    st.download_button(
                        "📊 Download CSV", 
                        data=csv_bytes, 
                        file_name="extracted_table.csv", 
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    xlsx_bytes = to_excel_download_rich_superscripts(stitched_df)
                    st.download_button(
                        "📈 Download Excel", 
                        data=xlsx_bytes, 
                        file_name="extracted_table.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                with col3:
                    if "enhanced_report" in st.session_state:
                        st.download_button(
                            "📄 Download Report", 
                            data=st.session_state["enhanced_report"].encode("utf-8"), 
                            file_name="clinical_trial_report.md", 
                            mime="text/markdown",
                            use_container_width=True
                        )

                # Rule-based analysis with enhanced styling
                st.markdown("---")
                st.markdown("## 🧪 Clinical Trial Analysis Results")
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                # Period Analysis
                with analysis_col1:
                    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                    try:
                        period_row_index, visit_row_index = find_period_and_visit_rows(stitched_df)
                        
                        if period_row_index is not None and visit_row_index is not None:
                            r0 = stitched_df.iloc[period_row_index, 1:].tolist()
                            r1 = stitched_df.iloc[visit_row_index, 1:].tolist()
                            r0 = [str(x).replace("\n", " ").strip() for x in r0]
                            r1 = [str(x).replace("\n", " ").strip() for x in r1]
                            
                            period_map = analyze_period(r0, r1)
                            period_items = [
                                {"Period": p, "Visit Count": len(v), "Visits": ", ".join(normalize_visit(x) for x in v)}
                                for p, v in period_map.items()
                            ]
                            period_df = pd.DataFrame(period_items)
                            
                            st.markdown("#### 📊 Period Analysis")
                            st.dataframe(period_df, use_container_width=True, hide_index=True)
                            st.success(f"✅ Found {len(period_items)} periods")
                        else:
                            st.markdown("#### 📊 Period Analysis")
                            st.warning("⚠️ Period or visit rows not detected")
                            
                    except Exception as e:
                        st.markdown("#### 📊 Period Analysis")
                        st.error(f"❌ Analysis failed: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Visit Mapping Analysis  
                with analysis_col2:
                    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                    try:
                        if visit_row_index is not None:
                            dynamic_start_row = find_first_data_row_for_mapping(stitched_df, visit_row_index)
                            mapping = analyze_row_filled_columns(stitched_df, start_row=dynamic_start_row, visit_row=visit_row_index)
                            
                            formatted_output = [
                                {"Sample": row_name, "Visits": ", ".join(normalize_visit(i) for i in data["visits"]), "Count": data["count"]}
                                for row_name, data in mapping.items()
                            ]
                            map_df = pd.DataFrame(formatted_output)
                            
                            st.markdown("#### 🗺️ Visit Mapping")
                            st.dataframe(map_df, use_container_width=True, hide_index=True)
                            st.success(f"✅ Mapped {len(map_df)} samples")
                        else:
                            st.markdown("#### 🗺️ Visit Mapping")
                            st.warning("⚠️ Visit row not detected")
                            
                    except Exception as e:
                        st.markdown("#### 🗺️ Visit Mapping")
                        st.error(f"❌ Analysis failed: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Visit Count Analysis (Full width)
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                try:
                    if visit_row_index is not None:
                        dot_res = countdots(stitched_df, visit_row=visit_row_index)
                        
                        count_output = [
                            {"Visit": visit, "Count": info["count"], "Rows": ", ".join(info["rows"])}
                            for visit, info in dot_res.items()
                        ]
                        count_df = pd.DataFrame(count_output)
                        
                        st.markdown("#### 🔢 Visit Count Analysis")
                        st.dataframe(count_df, use_container_width=True, hide_index=True)
                        
                        # Summary metrics
                        total_collections = sum([info["count"] for info in dot_res.values()])
                        st.metric("Total Collections", f"{total_collections:,}")
                    else:
                        st.markdown("#### 🔢 Visit Count Analysis")
                        st.warning("⚠️ Visit row not detected")
                        
                except Exception as e:
                    st.markdown("#### 🔢 Visit Count Analysis")
                    st.error(f"❌ Analysis failed: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

                # Annotations (Full width)
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                try:
                    superscripts_list = unique_superscripts_list(stitched_df)
                    processed_list = process_annotations(superscripts_list)
                    
                    st.markdown("#### 📝 Extracted Annotations")
                    if processed_list:
                        annotation_cols = st.columns(min(3, len(processed_list)))
                        for i, sup in enumerate(processed_list):
                            col_idx = i % len(annotation_cols)
                            with annotation_cols[col_idx]:
                                if MEANING_OVERRIDES and sup in MEANING_OVERRIDES:
                                    meaning = MEANING_OVERRIDES.get(sup)
                                    st.info(f"**{sup}**: {meaning}")
                                else:
                                    st.info(f"**{sup}**")
                        st.success(f"✅ Found {len(processed_list)} annotations")
                    else:
                        st.warning("⚠️ No annotations detected")
                        
                except Exception as e:
                    st.error(f"❌ Annotation extraction failed: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2: RAG Analysis
    with tab2:
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2>🧠 Advanced RAG Analysis</h2>
            <p>AI-powered insights combining extracted table data with PDF content</p>
        </div>
        """, unsafe_allow_html=True)
        
        if "extracted_df" not in st.session_state:
            st.markdown("""
            <div style="background: #ffc107; color: #212529; padding: 1.5rem; border-radius: 8px; text-align: center;">
                <h3>⚠️ No Table Data Available</h3>
                <p>Please extract a table first in the <strong>Table Extraction & Analysis</strong> tab</p>
                <p><small>The RAG analysis requires extracted table data to function</small></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # RAG Controls Section
            st.markdown("### 🎛️ Analysis Controls")
            rag_col1, rag_col2 = st.columns([2, 1])
            
            with rag_col1:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                if st.button("🚀 Build RAG Index & Generate Report", type="primary", use_container_width=True):
                    try:
                        with st.spinner("🔧 Initializing AI model..."):
                            model, temp, resolved_name = resolve_and_init_gemini(temperature=temperature)
                            
                        st.markdown(f"""
                        <div style="background: #28a745; color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                            <h4>🤖 Model Initialized</h4>
                            <p><strong>Model:</strong> {resolved_name}</p>
                            <p><strong>Temperature:</strong> {temperature}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.spinner("🔍 Building RAG index with table analysis..."):
                            rag = EnhancedRAGEngine(model=model, temperature=temp, embed_model=EMBED_MODEL)
                            rag.ingest_with_analysis(
                                df=st.session_state["extracted_df"], 
                                pdf_path=st.session_state["pdf_path"], 
                                table_name="extracted_table"
                            )
                            
                        st.success("✅ RAG index built successfully with table analysis integration")
                            
                        with st.spinner("🧠 Generating enhanced analysis report..."):
                            report_md = rag.generate_enhanced_report(default_prompt)
                            st.session_state["enhanced_report"] = report_md
                            
                    except Exception as e:
                        if is_resource_exhausted(e):
                            st.error("🚫 API quota exceeded. Please check your usage limits.")
                            st.info("💡 Consider using a lower creativity setting or reducing context size")
                        else:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            st.exception(e)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with rag_col2:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.text_area("Custom Analysis Prompt", value=default_prompt, height=150, key="custom_prompt", 
                           help="Customize the AI analysis prompt to focus on specific aspects")
                
                if st.button("🔄 Regenerate with Custom Prompt", use_container_width=True):
                    if "extracted_df" in st.session_state:
                        try:
                            with st.spinner("Regenerating with custom prompt..."):
                                model, temp, _ = resolve_and_init_gemini(temperature=temperature)
                                rag = EnhancedRAGEngine(model=model, temperature=temp)
                                rag.ingest_with_analysis(
                                    df=st.session_state["extracted_df"], 
                                    pdf_path=st.session_state["pdf_path"]
                                )
                                
                                custom_report = rag.generate_enhanced_report(st.session_state["custom_prompt"])
                                st.session_state["enhanced_report"] = custom_report
                                
                            st.success("✅ Custom report generated!")
                            
                        except Exception as e:
                            st.error(f"❌ Custom analysis failed: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Display report with enhanced styling
            if "enhanced_report" in st.session_state:
                st.markdown("---")
                st.markdown("### 📊 Generated Analysis Report")
                
                # Report container with styling
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown(st.session_state["enhanced_report"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download options
                report_col1, report_col2, report_col3 = st.columns(3)
                with report_col1:
                    st.download_button(
                        "📄 Download Report (MD)",
                        data=st.session_state["enhanced_report"].encode("utf-8"),
                        file_name="clinical_trial_enhanced_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with report_col2:
                    # Convert to HTML for better formatting
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Clinical Trial Analysis Report</title>
                        <meta charset="UTF-8">
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 2rem; line-height: 1.6; }}
                            table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #f2f2f2; }}
                            h1, h2, h3 {{ color: #333; }}
                        </style>
                    </head>
                    <body>
                        {st.session_state["enhanced_report"]}
                    </body>
                    </html>
                    """
                    st.download_button(
                        "🌐 Download Report (HTML)",
                        data=html_content.encode("utf-8"),
                        file_name="clinical_trial_enhanced_report.html",
                        mime="text/html",
                        use_container_width=True
                    )
                with report_col3:
                    st.info("💡 More formats coming soon!")

            # Quick insights if available
            if "extracted_df" in st.session_state:
                st.markdown("---")
                st.markdown("### 🔍 Quick Data Insights")
                insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
                
                df = st.session_state["extracted_df"]
                with insight_col1:
                    st.metric("Table Size", f"{df.shape[0]} × {df.shape[1]}")
                with insight_col2:
                    try:
                        _, visit_row_idx = find_period_and_visit_rows(df)
                        if visit_row_idx is not None:
                            visits = df.iloc[visit_row_idx, 1:].dropna()
                            st.metric("Visit Columns", len(visits))
                        else:
                            st.metric("Visit Columns", "N/A")
                    except:
                        st.metric("Visit Columns", "Error")
                with insight_col3:
                    non_null_cells = df.notna().sum().sum()
                    total_cells = df.shape[0] * df.shape[1]
                    fill_rate = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
                    st.metric("Data Fill Rate", f"{fill_rate:.1f}%")
                with insight_col4:
                    if "enhanced_report" in st.session_state:
                        word_count = len(st.session_state["enhanced_report"].split())
                        st.metric("Report Words", f"{word_count:,}")

    # Tab 3: Debug Info
    with tab3:
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2>🐛 Debug Information</h2>
            <p>Technical details and troubleshooting information</p>
        </div>
        """, unsafe_allow_html=True)
        
        if "extracted_df" in st.session_state:
            df = st.session_state["extracted_df"]
            
            # Debug sections with enhanced styling
            debug_col1, debug_col2 = st.columns(2)
            
            with debug_col1:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("#### 🔍 Rule Detection Results")
                try:
                    period_row_index, visit_row_index = find_period_and_visit_rows(df)
                    
                    debug_data = {
                        "Metric": ["Period Row Index", "Visit Row Index", "Dynamic Start Row", "DataFrame Shape", "Column Count"],
                        "Value": [
                            period_row_index if period_row_index is not None else "Not Found",
                            visit_row_index if visit_row_index is not None else "Not Found",
                            find_first_data_row_for_mapping(df, visit_row_index) if visit_row_index is not None else "N/A",
                            f"{df.shape[0]} × {df.shape[1]}",
                            len(df.columns)
                        ]
                    }
                    debug_df = pd.DataFrame(debug_data)
                    st.dataframe(debug_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"❌ Debug analysis failed: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with debug_col2:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("#### 📊 Data Quality Metrics")
                
                # Calculate data quality metrics
                null_count = df.isnull().sum().sum()
                total_cells = df.shape[0] * df.shape[1]
                fill_rate = ((total_cells - null_count) / total_cells * 100) if total_cells > 0 else 0
                
                quality_data = {
                    "Metric": ["Total Cells", "Null Cells", "Fill Rate", "Empty Columns", "Duplicate Rows"],
                    "Value": [
                        f"{total_cells:,}",
                        f"{null_count:,}",
                        f"{fill_rate:.1f}%",
                        len([col for col in df.columns if df[col].isnull().all()]),
                        len(df) - len(df.drop_duplicates())
                    ]
                }
                quality_df = pd.DataFrame(quality_data)
                st.dataframe(quality_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Full table preview
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("#### 👀 Table Preview (First 10 Rows)")
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Column analysis
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("#### 📋 Column Analysis")
            
            col_analysis = []
            for i, col in enumerate(df.columns):
                col_data = df[col]
                unique_count = col_data.nunique()
                null_count = col_data.isnull().sum()
                
                col_analysis.append({
                    "Column Index": i,
                    "Column Name": col if col else f"Col_{i}",
                    "Data Type": str(col_data.dtype),
                    "Unique Values": unique_count,
                    "Null Count": null_count,
                    "Fill Rate": f"{((len(col_data) - null_count) / len(col_data) * 100):.1f}%" if len(col_data) > 0 else "0%"
                })
            
            col_df = pd.DataFrame(col_analysis)
            st.dataframe(col_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Session state info
            if st.checkbox("🔍 Show Session State Debug", help="Technical information about stored data"):
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("#### 🗄️ Session State Information")
                session_info = {
                    "Key": [],
                    "Type": [],
                    "Size/Info": []
                }
                
                for key, value in st.session_state.items():
                    session_info["Key"].append(key)
                    session_info["Type"].append(type(value).__name__)
                    if hasattr(value, 'shape'):
                        session_info["Size/Info"].append(f"Shape: {value.shape}")
                    elif hasattr(value, '__len__') and not isinstance(value, str):
                        session_info["Size/Info"].append(f"Length: {len(value)}")
                    else:
                        session_info["Size/Info"].append("Scalar/Object")
                
                session_df = pd.DataFrame(session_info)
                st.dataframe(session_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #6c757d; color: white; padding: 1.5rem; border-radius: 8px; text-align: center;">
                <h3>📭 No Debug Data Available</h3>
                <p>Extract a table first to see debug information</p>
                <p><small>Debug information will include rule detection results, data quality metrics, and technical details</small></p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Enhanced landing page when no file is uploaded
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0;">
        <h2>🚀 Getting Started</h2>
        <p>Follow these steps to analyze your clinical trial protocol:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📤 1. Upload</h3>
            <p>Upload your clinical trial protocol PDF using the file uploader above</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 2. Extract</h3>
            <p>Advanced table detection extracts visit schedules and protocol data automatically</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 3. Analyze</h3>
            <p>AI-powered analysis generates comprehensive reports combining table data with PDF content</p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 2rem;">
    <h4>🧬 Clinical Trial Analytics Platform</h4>
    <p style="margin: 0.5rem 0;">Powered by advanced OCR, rule-based analysis, and AI-driven insights</p>
    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
        Built with Streamlit • Camelot OCR • Gemini AI • Advanced RAG Technology
    </p>
</div>
""", unsafe_allow_html=True)