# Clinical PDF Data Extraction Platform

## Overview

A professional web application for extracting, analyzing, and visualizing clinical trial data from PDF documents. The platform combines AI-powered OCR capabilities with advanced analytics to transform complex medical PDFs into structured, searchable data tables. Built for precision and efficiency in handling clinical study documents.

**Core Value Proposition**: Automated extraction of tabular data from clinical PDFs (both vector-based and scanned documents) with built-in analytics, visualization, and export capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework**: React 18 with TypeScript, built using Vite as the build tool and development server.

**UI Component System**: Shadcn UI (Radix UI primitives) + Tailwind CSS
- **Rationale**: Professional, accessible components suitable for enterprise data applications. Provides clean, trustworthy aesthetic appropriate for medical/clinical contexts.
- **Design Philosophy**: Data-first approach prioritizing information clarity over decoration, WCAG AA accessibility compliance.
- **Styling Strategy**: Tailwind utility classes with custom design tokens for medical aesthetic. Typography uses Inter for UI/data and JetBrains Mono for monospace content.

**State Management**: TanStack Query (React Query) for server state management
- **Rationale**: Handles data fetching, caching, and synchronization without additional global state libraries.
- **Configuration**: Infinite stale time, no automatic refetching (user-controlled data updates).

**Routing**: Wouter (lightweight React router)
- **Key Routes**: Landing page, Dashboard, Upload, Tables view, Analytics, Export functionality.

**Key Pages**:
- **Landing**: Marketing/introduction page
- **Dashboard**: Overview statistics (total extractions, success/failure counts, recent uploads) + Clinical Study Metrics (subjects, visits, items, collections, study periods, annotations)
- **Upload**: Drag-and-drop PDF upload with progress tracking
- **Tables**: View and search extracted tabular data
- **Analytics**: Visualizations using Recharts (bar charts, frequency analysis)
- **Export**: Download extracted data as CSV, Excel, or JSON

### Backend Architecture

**Hybrid Architecture**: Node.js (Express) + Python (FastAPI)

**Node.js Server (Express)**:
- Primary application server handling frontend serving, routing, and proxying
- **Port**: Production deployment handles both static assets and API routing
- **Key Responsibilities**:
  - Vite dev server integration in development mode
  - Static file serving in production
  - Request logging middleware
  - File upload handling (multer for multipart form data)
  - Proxying PDF processing requests to Python backend

**Python Server (FastAPI)**:
- **Port**: 8000
- **Purpose**: CPU-intensive PDF processing and OCR operations
- **Spawned Process**: Node.js spawns Python server as child process on startup
- **Key Responsibilities**:
  - PDF table extraction using Camelot (vector PDFs)
  - OCR processing using Groq API with Mistral models (scanned PDFs)
  - Table stitching across multiple pages
  - Data normalization and cleanup
  - Analytics computation

**Processing Strategy**:
1. Detect if PDF is vector-based (Camelot can parse) or requires OCR
2. Vector PDFs: Use Camelot with both "lattice" and "stream" flavors
3. Scanned PDFs: Convert to images, process with Groq Vision API
4. Post-processing: Stitch multi-page tables, normalize headers, clean null values
5. Analytics generation: Visit frequency analysis, period analysis, summary statistics

### Data Storage

**Current Implementation**: In-memory storage (MemStorage class)
- **Structure**: Map-based storage with UUID keys
- **Data Model**: PdfExtraction records containing filename, upload timestamp, processing status, extracted data, analytics data, and error messages

**Database Schema Preparation**: Drizzle ORM with PostgreSQL schema defined
- **Table**: `pdf_extractions` with JSONB columns for flexible data storage
- **Migration Ready**: Schema and Drizzle config present but not currently connected
- **Future Integration**: Application structured to swap MemStorage for database-backed storage without API changes

**Key Design Decision**: 
- **Problem**: Need flexible storage for variable table structures
- **Solution**: JSONB columns for `extractedData` and `analyticsData`
- **Rationale**: Clinical tables vary significantly in structure; JSONB provides flexibility while maintaining queryability

### API Design

**RESTful Endpoints**:
- `POST /api/upload` - Upload PDF file, returns extraction ID
- `GET /api/extractions` - List all extractions
- `GET /api/extractions/:id` - Get specific extraction with data
- `GET /api/dashboard` - Dashboard statistics
- `GET /api/export/:id?format=csv|excel|json` - Download extracted data

**Data Flow**:
1. Frontend uploads PDF via multipart form data
2. Express receives file, creates extraction record (status: "processing")
3. Express forwards file to Python FastAPI server
4. Python processes PDF asynchronously, returns results
5. Express updates extraction record with results or error
6. Frontend polls/refetches to get updated status

## External Dependencies

### Third-Party Services

**Groq API**: AI-powered OCR for scanned PDFs
- **Purpose**: Vision model (Mistral) for extracting tables from PDF images
- **Configuration**: Requires `GROQ_API_KEY` environment variable
- **Usage Pattern**: Convert PDF pages to base64 images, send to Groq for structured table extraction

### Database

**Neon Database** (PostgreSQL via @neondatabase/serverless)
- **Current Status**: Schema defined, not actively used (in-memory storage active)
- **Connection**: Expects `DATABASE_URL` environment variable
- **ORM**: Drizzle ORM for type-safe database operations
- **Session Storage**: `connect-pg-simple` for PostgreSQL session storage (prepared but not implemented)

### Python Libraries

**Core Processing**:
- **pdf2image**: Convert PDF pages to images for OCR path
- **camelot-py**: Extract tables from vector-based PDFs
- **PyPDF2**: PDF manipulation utilities
- **pandas**: Data manipulation and table operations
- **numpy**: Numerical operations for data processing

**API Framework**:
- **FastAPI**: Python web framework for processing API
- **uvicorn**: ASGI server for FastAPI

**External APIs**:
- **Groq Python SDK**: Interface to Groq Vision API

### Frontend Libraries

**UI Components**: Complete Radix UI ecosystem (@radix-ui/react-*)
- Accessible primitives for dialogs, dropdowns, tooltips, navigation, etc.

**Data Visualization**: Recharts for clinical data charts

**Form Handling**: React Hook Form with Zod validation (@hookform/resolvers)

**File Upload**: react-dropzone for drag-and-drop PDF uploads

**Utilities**:
- date-fns: Date formatting and manipulation
- clsx + tailwind-merge: Conditional CSS class management
- class-variance-authority: Component variant system

### Build Tools

**Vite**: Build tool and dev server with React plugin

**TypeScript**: Type safety across entire application

**PostCSS**: CSS processing with Tailwind and Autoprefixer

**ESBuild**: Server-side bundling for production

### Development Tools

**Replit-Specific**:
- @replit/vite-plugin-runtime-error-modal: Error overlay
- @replit/vite-plugin-cartographer: Code mapping
- @replit/vite-plugin-dev-banner: Development banner

## Replit Environment Setup

### Environment Configuration

**Required Secrets**:
- `GROQ_API_KEY`: API key for Groq Vision API (used for OCR processing of scanned PDFs)

### Running the Application

**Development Mode**:
- The "Start application" workflow runs `npm run dev` which:
  1. Starts the Node.js Express server on port 5000 (frontend + API proxy)
  2. Automatically spawns the Python FastAPI server on port 8000 (PDF processing)
  3. Vite dev server provides hot module replacement for frontend development

**Python Environment**:
- Python dependencies are managed using `uv` (modern Python package manager)
- Virtual environment is created at `.pythonlibs/` directory
- The Node.js server spawns the Python process using `.pythonlibs/bin/python`

**Port Configuration**:
- Frontend/API: Port 5000 (0.0.0.0, publicly accessible)
- Python Backend: Port 8000 (localhost only, accessed via Node.js proxy)

### Deployment Configuration

**Deployment Type**: VM (Virtual Machine)
- **Rationale**: Application requires persistent Python FastAPI server process
- **Build Command**: `npm run build` (compiles TypeScript + builds Vite frontend)
- **Run Command**: `npm start` (production mode with pre-built assets)

### Recent Setup Actions

**November 19, 2025**:
1. ✅ **Added Clinical Study Metrics to Dashboard**:
   - Created `ClinicalMetrics` interface in `shared/schema.ts` with type safety
   - Implemented consolidated `/api/clinical-metrics` FastAPI endpoint (single request for all 7 metrics)
   - Updated dashboard to display: Total Subjects (90), Total Visits (18), Total Items (29), Total Collections (1620), Collections Pre-Study (116), Study Periods (4), and Annotations (11)
   - Highlighted Annotations metric in purple/primary themed card above metrics grid
   - Maintained backward compatibility with legacy individual endpoints
   - Added clear documentation that metrics are static values per user specification

**November 18, 2025**:
1. ✅ Installed Node.js dependencies via `npm install`
2. ✅ Installed Python dependencies to `.pythonlibs/lib/python3.11/site-packages` using pip with PYTHONUSERBASE
3. ✅ Configured `server/routes.ts` to spawn Python server with system python3
4. ✅ Configured workflow for development server on port 5000
5. ✅ Configured deployment settings (VM deployment target)
6. ✅ Updated `.gitignore` to exclude Python virtual environment and cache files
7. ✅ **Enhanced UI Features**:
   - Added "Home" navigation option at top of sidebar
   - Implemented AppContext for persistent state management with localStorage
   - Enhanced Tables page with column sorting, striped rows, and CSV export
   - Updated Analytics and Export pages with persistent state integration
   - Improved chatbot system prompt for concise responses
   - Added validation logic to prevent stale extraction IDs with loading guards
8. ✅ Verified application functionality: All pages working with both Node.js and Python servers running correctly

### Python Package Installation

Python packages are installed to `.pythonlibs/lib/python3.11/site-packages` which is automatically included in Python's sys.path. To install additional packages:

```bash
PYTHONUSERBASE=/home/runner/workspace/.pythonlibs python3 -m pip install --user <package-name>
```

Current installed packages: fastapi, uvicorn, python-multipart, groq, pypdf2, pdf2image, camelot-py, pandas, numpy, pillow, python-dotenv, openpyxl