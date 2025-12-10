import os
import io
import json
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

# Import extraction utilities
from utils_ocr import (
    convert_pdf_to_images,
    is_vector_pdf,
    extract_tables_with_camelot,
    extract_tables_with_mistral_ocr,
    MEANING_OVERRIDES,
)
from table_postprocess import (
    stitch_multipage_tables,
    normalize_headers_with_subcolumns,
    drop_empty_cols_and_fix_nulls,
    to_csv_download,
    to_excel_download,
)

app = FastAPI(title="Clinical PDF Extractor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo (matches TypeScript storage)
extractions_db = {}

# Helper functions
def generate_analytics(df: pd.DataFrame) -> dict:
    """Generate analytics data from extracted dataframe."""
    try:
        if df is None or df.empty:
            return None
        
        analytics = {
            "visitFrequency": [],
            "periodAnalysis": [],
            "assessmentsByVisit": [],
            "meaningOverrides": [{"key": k, "description": v} for k, v in MEANING_OVERRIDES.items()],
            "totalVisits": 0,
            "totalAssessments": 0,
        }
        
        # Assuming visit row is at index 4 and period row at index 0 (based on your Streamlit code)
        if df.shape[0] > 4:
            visit_row_index = 4
            visit_names = df.iloc[visit_row_index, 1:].astype(str).str.strip().tolist()
            visit_names = [v for v in visit_names if v and v not in ["nan", "None", ""]]
            analytics["totalVisits"] = len(visit_names)
            
            # Count assessments per visit
            visit_frequency = {}
            for col_idx in range(1, df.shape[1]):
                if col_idx - 1 < len(visit_names):
                    visit = visit_names[col_idx - 1]
                    count = 0
                    for row_idx in range(visit_row_index + 1, df.shape[0]):
                        try:
                            val = str(df.iloc[row_idx, col_idx]).strip()
                            if val not in ["", "nan", "None", "none"]:
                                count += 1
                        except:
                            pass
                    visit_frequency[visit] = count
            
            analytics["visitFrequency"] = [
                {"visit": k, "count": v} for k, v in visit_frequency.items()
            ]
            
            # Period analysis
            if df.shape[0] > 0:
                period_row = df.iloc[0, 1:].astype(str).str.strip().tolist()
                period_map = {}
                for i, period in enumerate(period_row):
                    if i < len(visit_names) and period and period not in ["nan", "None", ""]:
                        if period not in period_map:
                            period_map[period] = []
                        period_map[period].append(visit_names[i])
                
                analytics["periodAnalysis"] = [
                    {"period": k, "visits": v} for k, v in period_map.items()
                ]
            
            # Assessments by visit
            assessments_by_visit = []
            for col_idx in range(1, df.shape[1]):
                if col_idx - 1 < len(visit_names):
                    visit = visit_names[col_idx - 1]
                    assessments = []
                    for row_idx in range(visit_row_index + 1, df.shape[0]):
                        try:
                            val = str(df.iloc[row_idx, col_idx]).strip()
                            if val not in ["", "nan", "None", "none"]:
                                row_name = str(df.iloc[row_idx, 0]).strip()
                                if row_name and row_name not in ["", "nan", "None"]:
                                    assessments.append(row_name)
                        except:
                            pass
                    
                    if assessments:
                        assessments_by_visit.append({
                            "visit": visit,
                            "assessments": assessments,
                            "count": len(assessments)
                        })
            
            analytics["assessmentsByVisit"] = assessments_by_visit
            analytics["totalAssessments"] = df.shape[0] - visit_row_index - 1
        
        return analytics
    except Exception as e:
        print(f"Analytics generation error: {e}")
        return None


@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard statistics."""
    all_extractions = list(extractions_db.values())
    
    total = len(all_extractions)
    successful = len([e for e in all_extractions if e["status"] == "completed"])
    failed = len([e for e in all_extractions if e["status"] == "failed"])
    processing = len([e for e in all_extractions if e["status"] == "processing"])
    success_rate = (successful / total * 100) if total > 0 else 0
    
    # Get recent extractions (last 5)
    recent = sorted(all_extractions, key=lambda x: x["uploadedAt"], reverse=True)[:5]
    
    return {
        "totalExtractions": total,
        "successfulExtractions": successful,
        "failedExtractions": failed,
        "processingExtractions": processing,
        "successRate": success_rate,
        "recentExtractions": recent,
    }


# NEW: Consolidated clinical metrics endpoint for dashboard
@app.get("/api/clinical-metrics")
async def get_clinical_metrics():
    """
    Get all clinical study metrics in a single request.
    
    NOTE: These are currently fixed values per user specification.
    Future enhancement: derive from actual extraction data.
    """
    # Static values as specified by user requirements
    num_subjects = 90
    num_visits = 18  # Also used for dynamic collection calculation
    
    return {
        "totalSubjects": num_subjects,
        "totalVisits": num_visits,
        "totalItems": 29,
        "totalCollections": num_subjects * num_visits,  # 90 * 18 = 1620
        "collectionsPreStudy": 116,
        "studyPeriods": 4,
        "annotations": 11
    }


# Legacy individual endpoints (kept for backwards compatibility)
@app.get("/api/subjects/count")
async def get_subjects_count():
    """Get total subjects count."""
    return {"count": 90}


@app.get("/api/visits/total")
async def get_visits_total():
    """Get total visits."""
    return {"total": 18}


@app.get("/api/items/total")
async def get_items_total():
    """Get total items."""
    return {"total": 29}


@app.get("/api/collections/total")
async def get_collections_total():
    """Get total collections (90 * number of dots - dynamic calculation)."""
    num_dots = 18
    return {"total": 90 * num_dots}


@app.get("/api/collections/prestudy")
async def get_collections_prestudy():
    """Get collections pre-study count."""
    return {"count": 116}


@app.get("/api/studyperiods/total")
async def get_study_periods():
    """Get total study periods."""
    return {"total": 4}


@app.get("/api/annotations/total")
async def get_annotations_total():
    """Get total annotations."""
    return {"total": 11}


@app.get("/api/extractions")
async def get_all_extractions():
    """Get all extractions."""
    all_extractions = list(extractions_db.values())
    return sorted(all_extractions, key=lambda x: x["uploadedAt"], reverse=True)


@app.get("/api/extractions/{extraction_id}")
async def get_extraction(extraction_id: str):
    """Get a specific extraction by ID."""
    if extraction_id not in extractions_db:
        raise HTTPException(status_code=404, detail="Extraction not found")
    return extractions_db[extraction_id]


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate unique ID
    import uuid
    extraction_id = str(uuid.uuid4())
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_pdf_path = tmp_file.name
    
    # Create initial extraction record
    from datetime import datetime
    extraction = {
        "id": extraction_id,
        "filename": file.filename,
        "uploadedAt": datetime.utcnow().isoformat(),
        "status": "processing",
        "extractedData": None,
        "analyticsData": None,
        "errorMessage": None,
    }
    extractions_db[extraction_id] = extraction
    
    # Process PDF asynchronously (simplified for demo - in production use background tasks)
    try:
        # Detect PDF type
        vector_flag = is_vector_pdf(tmp_pdf_path)
        
        # Extract tables
        if vector_flag:
            raw_tables = extract_tables_with_camelot(tmp_pdf_path, max_pages=50)
            if len(raw_tables) == 0:
                # Fallback to OCR
                images = convert_pdf_to_images(tmp_pdf_path, dpi=300, max_pages=50)
                groq_api_key = os.getenv("GROQ_API_KEY")
                raw_tables = extract_tables_with_mistral_ocr(
                    images=images,
                    api_key=groq_api_key,
                    model_name="mistral-ocr",
                    table_hint="Document Attribute Matrix; if multiple tables exist, focus on the matrix that spans pages.",
                )
                extraction_method = "ocr"
            else:
                extraction_method = "camelot"
        else:
            images = convert_pdf_to_images(tmp_pdf_path, dpi=300, max_pages=50)
            groq_api_key = os.getenv("GROQ_API_KEY")
            raw_tables = extract_tables_with_mistral_ocr(
                images=images,
                api_key=groq_api_key,
                model_name="mistral-ocr",
                table_hint="Document Attribute Matrix; if multiple tables exist, focus on the matrix that spans pages.",
            )
            extraction_method = "ocr"
        
        # Stitch and clean tables
        stitched_df = stitch_multipage_tables(raw_tables)
        if not stitched_df.empty:
            stitched_df = normalize_headers_with_subcolumns(stitched_df)
            stitched_df = drop_empty_cols_and_fix_nulls(stitched_df)
        
        # Convert to JSON-serializable format
        if not stitched_df.empty:
            headers = stitched_df.columns.tolist()
            rows = stitched_df.values.tolist()
            
            extraction["extractedData"] = {
                "headers": headers,
                "rows": rows,
                "metadata": {
                    "totalRows": len(rows),
                    "totalColumns": len(headers),
                    "extractionMethod": extraction_method,
                }
            }
            
            # Generate analytics
            analytics_data = generate_analytics(stitched_df)
            if analytics_data:
                extraction["analyticsData"] = analytics_data
        
        extraction["status"] = "completed"
        
    except Exception as e:
        extraction["status"] = "failed"
        extraction["errorMessage"] = str(e)
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_pdf_path)
        except:
            pass
    
    extractions_db[extraction_id] = extraction
    
    return {
        "id": extraction_id,
        "filename": file.filename,
        "status": extraction["status"],
        "message": "PDF uploaded and processed successfully" if extraction["status"] == "completed" else "Processing failed",
    }


@app.get("/api/export/{extraction_id}")
async def export_data(extraction_id: str, format: str = "csv"):
    """Export extraction data in specified format."""
    if extraction_id not in extractions_db:
        raise HTTPException(status_code=404, detail="Extraction not found")
    
    extraction = extractions_db[extraction_id]
    if not extraction.get("extractedData"):
        raise HTTPException(status_code=400, detail="No data available for export")
    
    # Recreate DataFrame
    headers = extraction["extractedData"]["headers"]
    rows = extraction["extractedData"]["rows"]
    df = pd.DataFrame(rows, columns=headers)
    
    if format == "csv":
        csv_bytes = to_csv_download(df)
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={extraction['filename']}.csv"}
        )
    elif format == "excel":
        xlsx_bytes = to_excel_download(df)
        return StreamingResponse(
            io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={extraction['filename']}.xlsx"}
        )
    elif format == "json":
        json_data = df.to_dict(orient="records")
        return JSONResponse(content=json_data)
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use csv, excel, or json")


class MessageEntry(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    conversationHistory: list[MessageEntry] = Field(default_factory=list)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chatbot conversations using Groq."""
    try:
        from groq import Groq
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=503,
                detail="AI service is not configured. GROQ_API_KEY is missing."
            )
        
        client = Groq(api_key=groq_api_key)
        user_message = request.message
        conversation_history = request.conversationHistory
        
        # System prompt for the PDF assistant - optimized for concise responses
        system_prompt = """You are a concise AI assistant for a Clinical PDF Data Extraction Platform. 

RESPONSE STYLE:
- Give direct, simple answers in 1-3 sentences
- No lengthy explanations unless specifically asked
- Use bullet points for lists (max 3-4 items)
- Avoid repetition and unnecessary details

HELP WITH:
- Upload/process PDFs
- Extracted data & analytics
- Platform features
- Export options

If asked about specific data, say: "Check the Tables or Analytics page for details."
Be friendly but brief."""
        
        # Build messages for the API
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in conversation_history[-6:]:  # Keep last 6 messages for context
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        
        assistant_response = response.choices[0].message.content
        
        return JSONResponse(content={"response": assistant_response})
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat service error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
