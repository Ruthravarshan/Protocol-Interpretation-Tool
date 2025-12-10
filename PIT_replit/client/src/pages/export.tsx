import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Download, FileText } from "lucide-react";
import { useState, useEffect } from "react";
import { useToast } from "@/hooks/use-toast";
import { useAppContext } from "@/contexts/AppContext";
import type { ExtractionResult } from "@shared/schema";

export default function ExportPage() {
  const { selectedExtractionId, setSelectedExtractionId } = useAppContext();
  const [exportFormat, setExportFormat] = useState<"csv" | "excel" | "json">("csv");
  const { toast } = useToast();

  const { data: extractions, isLoading } = useQuery<ExtractionResult[]>({
    queryKey: ["/api/extractions"],
  });

  const { data: selectedData } = useQuery<ExtractionResult>({
    queryKey: ["/api/extractions", selectedExtractionId],
    enabled: !!selectedExtractionId,
  });

  const completedExtractions = extractions?.filter(e => e.status === "completed") || [];

  useEffect(() => {
    if (isLoading) return;
    
    const isValidId = selectedExtractionId && completedExtractions.some(e => e.id === selectedExtractionId);
    
    if (!isValidId && completedExtractions.length > 0) {
      setSelectedExtractionId(completedExtractions[0].id);
    } else if (!isValidId && completedExtractions.length === 0) {
      setSelectedExtractionId("");
    }
  }, [isLoading, completedExtractions, selectedExtractionId, setSelectedExtractionId]);

  const handleExport = async () => {
    if (!selectedExtractionId) {
      toast({
        title: "No Extraction Selected",
        description: "Please select an extraction to export",
        variant: "destructive",
      });
      return;
    }

    try {
      const response = await fetch(`/api/export/${selectedExtractionId}?format=${exportFormat}`);
      
      if (!response.ok) {
        throw new Error("Export failed");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      
      const extension = exportFormat === "excel" ? "xlsx" : exportFormat;
      a.download = `${selectedData?.filename || "export"}.${extension}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast({
        title: "Export Successful",
        description: `Downloaded ${selectedData?.filename}.${extension}`,
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "Failed to export data. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-semibold mb-2" data-testid="text-export-title">Export Data</h1>
        <p className="text-muted-foreground">
          Download extracted table data in various formats
        </p>
      </div>

      {/* Export Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Export Configuration</CardTitle>
          <CardDescription>Select extraction and format for export</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Select Extraction */}
          <div className="space-y-2">
            <Label htmlFor="extraction-select">Select Extraction</Label>
            <Select value={selectedExtractionId} onValueChange={setSelectedExtractionId}>
              <SelectTrigger id="extraction-select" data-testid="select-extraction-export">
                <SelectValue placeholder="Choose a completed extraction" />
              </SelectTrigger>
              <SelectContent>
                {isLoading ? (
                  <SelectItem value="loading" disabled>Loading...</SelectItem>
                ) : completedExtractions.length === 0 ? (
                  <SelectItem value="none" disabled>No completed extractions</SelectItem>
                ) : (
                  completedExtractions.map(extraction => (
                    <SelectItem key={extraction.id} value={extraction.id}>
                      {extraction.filename} ({new Date(extraction.uploadedAt).toLocaleDateString()})
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>

          {/* Select Format */}
          <div className="space-y-3">
            <Label>Export Format</Label>
            <RadioGroup value={exportFormat} onValueChange={(value: any) => setExportFormat(value)}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="csv" id="csv" data-testid="radio-csv" />
                <Label htmlFor="csv" className="cursor-pointer">
                  CSV - Comma-separated values (universal compatibility)
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="excel" id="excel" data-testid="radio-excel" />
                <Label htmlFor="excel" className="cursor-pointer">
                  Excel - Microsoft Excel format (.xlsx)
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="json" id="json" data-testid="radio-json" />
                <Label htmlFor="json" className="cursor-pointer">
                  JSON - Structured data format (for developers)
                </Label>
              </div>
            </RadioGroup>
          </div>

          {/* Export Button */}
          <Button
            onClick={handleExport}
            disabled={!selectedExtractionId}
            className="w-full"
            data-testid="button-download"
          >
            <Download className="h-4 w-4 mr-2" />
            Download {exportFormat.toUpperCase()}
          </Button>
        </CardContent>
      </Card>

      {/* Preview */}
      {selectedData && (
        <Card>
          <CardHeader>
            <CardTitle>Export Preview</CardTitle>
            <CardDescription>Data that will be exported</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-start gap-4 p-4 border rounded-lg">
              <FileText className="h-8 w-8 text-primary mt-1" />
              <div className="flex-1">
                <h4 className="font-medium mb-1">{selectedData.filename}</h4>
                <p className="text-sm text-muted-foreground">
                  {selectedData.extractedData?.metadata?.totalRows} rows × {selectedData.extractedData?.metadata?.totalColumns} columns
                </p>
                <p className="text-sm text-muted-foreground">
                  Uploaded: {new Date(selectedData.uploadedAt).toLocaleString()}
                </p>
              </div>
            </div>

            {selectedData.extractedData && (
              <div className="p-4 bg-muted/50 rounded-lg">
                <h5 className="font-medium mb-2">Table Summary</h5>
                <ul className="text-sm space-y-1 text-muted-foreground">
                  <li>• Headers: {selectedData.extractedData.headers.length} columns</li>
                  <li>• Rows: {selectedData.extractedData.rows.length} data rows</li>
                  <li>• Extraction Method: {selectedData.extractedData.metadata?.extractionMethod || "Unknown"}</li>
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Format Information */}
      <Card>
        <CardHeader>
          <CardTitle>Format Information</CardTitle>
          <CardDescription>Details about each export format</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <h4 className="font-medium">CSV (Comma-Separated Values)</h4>
            <p className="text-sm text-muted-foreground">
              Universal text format that works with Excel, Google Sheets, and most data tools.
              Best for simple data exchange and compatibility.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-medium">Excel (.xlsx)</h4>
            <p className="text-sm text-muted-foreground">
              Native Microsoft Excel format with formatting preservation.
              Best for business users and detailed analysis in Excel.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-medium">JSON (JavaScript Object Notation)</h4>
            <p className="text-sm text-muted-foreground">
              Structured data format ideal for programmatic access and API integration.
              Best for developers and automated workflows.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
