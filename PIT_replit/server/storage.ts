import { type PdfExtraction, type InsertPdfExtraction } from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  // PDF Extraction operations
  createExtraction(extraction: InsertPdfExtraction): Promise<PdfExtraction>;
  getExtraction(id: string): Promise<PdfExtraction | undefined>;
  getAllExtractions(): Promise<PdfExtraction[]>;
  updateExtraction(id: string, updates: Partial<InsertPdfExtraction>): Promise<PdfExtraction | undefined>;
  deleteExtraction(id: string): Promise<boolean>;
}

export class MemStorage implements IStorage {
  private extractions: Map<string, PdfExtraction>;

  constructor() {
    this.extractions = new Map();
  }

  async createExtraction(insertExtraction: InsertPdfExtraction): Promise<PdfExtraction> {
    const id = randomUUID();
    const extraction: PdfExtraction = {
      id,
      filename: insertExtraction.filename,
      uploadedAt: new Date(),
      status: insertExtraction.status || "processing",
      extractedData: insertExtraction.extractedData || null,
      analyticsData: insertExtraction.analyticsData || null,
      errorMessage: insertExtraction.errorMessage || null,
    };
    this.extractions.set(id, extraction);
    return extraction;
  }

  async getExtraction(id: string): Promise<PdfExtraction | undefined> {
    return this.extractions.get(id);
  }

  async getAllExtractions(): Promise<PdfExtraction[]> {
    return Array.from(this.extractions.values()).sort(
      (a, b) => new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime()
    );
  }

  async updateExtraction(
    id: string,
    updates: Partial<InsertPdfExtraction>
  ): Promise<PdfExtraction | undefined> {
    const existing = this.extractions.get(id);
    if (!existing) return undefined;

    const updated: PdfExtraction = {
      ...existing,
      ...updates,
      id, // ensure id doesn't change
      uploadedAt: existing.uploadedAt, // preserve original upload time
    };
    this.extractions.set(id, updated);
    return updated;
  }

  async deleteExtraction(id: string): Promise<boolean> {
    return this.extractions.delete(id);
  }
}

export const storage = new MemStorage();
