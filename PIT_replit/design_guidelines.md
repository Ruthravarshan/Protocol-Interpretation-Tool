# Design Guidelines: Clinical PDF Data Extraction Platform

## Design Approach

**System Selected**: Shadcn UI + Tailwind CSS
**Rationale**: Professional data application requiring clean, accessible components with enterprise-grade reliability. Material Design principles for clarity and hierarchy in complex medical data interfaces.

**Core Principles**:
- Data-first: Information clarity trumps decoration
- Professional medical aesthetic: Clean, trustworthy, clinical
- Efficiency: Minimize cognitive load for complex data analysis
- Accessibility: WCAG AA compliance for all interactive elements

---

## Typography System

**Font Families**:
- Primary: Inter (UI elements, body text, data tables)
- Monospace: JetBrains Mono (data values, code snippets, IDs)

**Hierarchy**:
- Page Titles: text-3xl font-semibold (30px)
- Section Headers: text-2xl font-semibold (24px)
- Subsections: text-xl font-medium (20px)
- Body/Labels: text-base font-normal (16px)
- Data Tables: text-sm (14px)
- Captions/Meta: text-xs text-muted-foreground (12px)

**Implementation**: Use consistent letter-spacing (tracking-tight for headings, tracking-normal for body) and line-height (leading-relaxed for readability in data-heavy contexts).

---

## Layout System

**Spacing Primitives**: Use Tailwind units of **2, 4, 6, 8, 12, 16** for consistent rhythm
- Component padding: p-4, p-6, p-8
- Section gaps: gap-6, gap-8
- Card spacing: space-y-4, space-y-6
- Page margins: mx-auto max-w-7xl px-6

**Grid Structure**:
- Dashboard: 12-column grid (grid-cols-12)
- Analytics: 2-3 column layout (lg:grid-cols-3 md:grid-cols-2)
- Tables: Full-width with internal column management
- Sidebar Navigation: Fixed 256px width (w-64)

**Viewport Management**:
- App chrome (header + nav): Fixed height ~64px + sidebar
- Content area: Scrollable regions with proper overflow handling
- No forced viewport heights - let content breathe naturally

---

## Component Library

### Navigation Architecture
**Side Navigation (Persistent)**:
- Fixed left sidebar (w-64, bg-card, border-r)
- Logo/brand at top (h-16, p-4)
- Navigation items with icons + labels
- Active state: bg-accent, text-accent-foreground, border-l-2
- Hover state: bg-accent/50
- Sections: Dashboard, Upload, Tables, Analytics, Export

**Top Header Bar**:
- User profile dropdown (right-aligned)
- Breadcrumb navigation (left-aligned)
- Processing status indicator (when active)
- Height: h-16, border-b

### Primary Components

**File Upload Zone**:
- Large dropzone: min-h-64, border-2 border-dashed rounded-lg
- Hover state: border-primary, bg-primary/5
- Active drop state: border-primary, bg-primary/10
- Icons: Upload cloud icon (h-12 w-12)
- Supporting text: file size limits, accepted formats
- Progress bar during processing (h-2, rounded-full)

**Data Tables**:
- Shadcn Table component with sorting, filtering
- Sticky header: position-sticky top-0 bg-background
- Row hover: hover:bg-muted/50
- Alternating rows: even:bg-muted/20 (optional for dense data)
- Action column: right-aligned with icon buttons
- Pagination controls: bottom-aligned, space-y-4

**Cards (Dashboard Metrics)**:
- Shadcn Card component (rounded-lg border bg-card)
- Header: title + optional icon
- Content: Large metric value (text-4xl font-bold) + subtitle
- Footer: Trend indicator or secondary metric
- Grid: 3-column on desktop (grid-cols-3 gap-6)

**Charts & Visualizations**:
- Use Recharts library for consistency
- Bar charts: Visit frequency, period analysis
- Timeline: Horizontal assessment schedule
- Height: Standard h-80 for primary charts, h-64 for secondary
- Legends: positioned top-right or bottom
- Tooltips: Show detailed data on hover
- Responsive: Adjust layout for mobile (stack vertically)

**Forms & Inputs**:
- Shadcn Form components (Input, Select, Checkbox)
- Label above input (text-sm font-medium mb-2)
- Helper text below (text-xs text-muted-foreground)
- Error state: border-destructive, text-destructive
- Focus ring: ring-2 ring-ring ring-offset-2

**Buttons**:
- Primary: bg-primary text-primary-foreground (actions)
- Secondary: bg-secondary text-secondary-foreground (cancel)
- Outline: border border-input (alternative actions)
- Ghost: transparent (icon buttons in tables)
- Sizes: h-10 px-4 (default), h-9 px-3 (small for tables)

**Status Indicators**:
- Processing badge: Animated pulse + spinner icon
- Success: bg-green-500/10 text-green-700 border-green-500/20
- Error: bg-destructive/10 text-destructive border-destructive/20
- Info: bg-blue-500/10 text-blue-700 border-blue-500/20

### Specialized Components

**Export Panel**:
- Format selection: Radio group (CSV, Excel, JSON)
- Preview table of selected data
- Download button (primary, full-width on mobile)
- Export history list (scrollable, max-h-96)

**Analytics Dashboard**:
- KPI cards grid (top section)
- Chart section (2-column: lg:grid-cols-2)
- Filter controls (sticky top bar within content)
- Comparison views (side-by-side tables/charts)

---

## Page Layouts

### Landing/Home Page
**Hero Section** (h-screen or min-h-96):
- Centered content: max-w-4xl mx-auto text-center
- Headline: text-5xl font-bold leading-tight
- Subheadline: text-xl text-muted-foreground mt-4
- Primary CTA: Large button (h-12 px-8)
- Hero image: Medical/clinical theme (abstract data visualization or clinical setting)
- Background: Subtle gradient (from-background to-muted)

**Features Section** (3-column grid):
- Icon + Title + Description cards
- Icons from Lucide (consistent style)
- Padding: py-24

**CTA Section**:
- bg-primary text-primary-foreground
- Centered content with action button
- Padding: py-16

### Dashboard View
**Layout**:
- Top: KPI cards (4 metrics: Total PDFs, Total Tables, Active Extractions, Success Rate)
- Middle: Recent activity table + Quick actions
- Bottom: Analytics preview chart

**Multi-column Strategy**:
- KPI cards: grid-cols-2 lg:grid-cols-4 gap-6
- Content area: lg:grid-cols-3 (2 cols content + 1 col sidebar)

### Upload View
**Single-column focused**:
- max-w-4xl mx-auto
- Dropzone (prominent)
- Upload queue list below
- Processing status cards (space-y-4)

### Tables View
**Full-width data table**:
- Filters/search bar (top, sticky)
- Table (responsive, horizontal scroll on mobile)
- Export actions (bottom toolbar)

### Analytics View
**Dashboard layout**:
- Filter controls (top bar)
- Charts grid: 2-column on desktop
- Summary statistics cards
- Comparison tables

---

## Visual Treatment

**Borders**: border (1px), rounded-lg (8px) for cards, rounded-md (6px) for inputs
**Shadows**: shadow-sm for cards, shadow-md for modals/popovers
**Spacing**: Consistent gap-6 between sections, gap-4 within components
**Transitions**: transition-colors duration-200 for hover states

---

## Animations

**Minimal, purposeful animations**:
- Loading spinners: animate-spin for processing states
- Data updates: Fade in new rows (animate-in fade-in duration-300)
- Page transitions: None (instant navigation for data app)
- Chart animations: Recharts default animations (subtle entry)

**Avoid**: Distracting parallax, scroll-triggered animations, decorative motion

---

## Images

**Hero Section**: 
- Large background image showing clinical research setting or abstract data visualization
- Overlay: Semi-transparent gradient for text readability
- Treatment: Subtle blur on background, sharp foreground elements

**Dashboard/App Views**:
- Empty state illustrations (when no data)
- Icon sets from Lucide (File, BarChart, Table, Download, Upload, etc.)
- No decorative imagery in functional views

---

## Accessibility Notes

- Maintain 4.5:1 contrast ratio for all text
- Focus indicators visible on all interactive elements
- ARIA labels for icon-only buttons
- Keyboard navigation support throughout
- Screen reader announcements for processing states and data updates