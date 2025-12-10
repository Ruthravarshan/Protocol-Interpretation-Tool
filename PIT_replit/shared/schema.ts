import { sql } from "drizzle-orm";
import { pgTable, text, varchar, jsonb, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// PDF Upload and Extraction Data Models
export const pdfExtractions = pgTable("pdf_extractions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  filename: text("filename").notNull(),
  uploadedAt: timestamp("uploaded_at").notNull().defaultNow(),
  status: text("status").notNull().default("processing"), // processing, completed, failed
  extractedData: jsonb("extracted_data"), // stores the extracted table data
  analyticsData: jsonb("analytics_data"), // stores computed analytics
  errorMessage: text("error_message"),
});

export const insertPdfExtractionSchema = createInsertSchema(pdfExtractions).pick({
  filename: true,
  status: true,
  extractedData: true,
  analyticsData: true,
  errorMessage: true,
});

export type InsertPdfExtraction = z.infer<typeof insertPdfExtractionSchema>;
export type PdfExtraction = typeof pdfExtractions.$inferSelect;

// TypeScript interfaces for API contracts
export interface UploadResponse {
  id: string;
  filename: string;
  status: string;
  message: string;
}

export interface ExtractionResult {
  id: string;
  filename: string;
  status: string;
  uploadedAt: string;
  extractedData?: {
    headers: string[];
    rows: any[][];
    metadata?: {
      totalRows: number;
      totalColumns: number;
      extractionMethod: string; // 'camelot' | 'ocr'
    };
  };
  analyticsData?: {
    visitFrequency: { visit: string; count: number }[];
    periodAnalysis: { period: string; visits: string[] }[];
    assessmentsByVisit: { visit: string; assessments: string[]; count: number }[];
    meaningOverrides: { key: string; description: string }[];
    totalVisits: number;
    totalAssessments: number;
  };
  errorMessage?: string;
}

export interface AnalyticsData {
  visitFrequency: { visit: string; count: number }[];
  periodAnalysis: { period: string; visits: string[] }[];
  assessmentsByVisit: { visit: string; assessments: string[]; count: number }[];
  meaningOverrides: { key: string; description: string }[];
  totalVisits: number;
  totalAssessments: number;
}

export interface ExportRequest {
  id: string;
  format: 'csv' | 'excel' | 'json';
}

export interface DashboardStats {
  totalExtractions: number;
  successfulExtractions: number;
  failedExtractions: number;
  processingExtractions: number;
  successRate: number;
  recentExtractions: ExtractionResult[];
}

// Clinical Metrics for Dashboard
// NOTE: These are currently fixed values per user specification.
// Future enhancement: derive from actual extraction data.
export interface ClinicalMetrics {
  totalSubjects: number;
  totalVisits: number;
  totalItems: number;
  totalCollections: number;
  collectionsPreStudy: number;
  studyPeriods: number;
  annotations: number;
}
