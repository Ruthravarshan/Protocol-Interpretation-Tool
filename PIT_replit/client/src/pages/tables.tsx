import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Download, Search, ArrowUpDown, FileDown } from "lucide-react";
import { useState, useEffect } from "react";
import { useAppContext } from "@/contexts/AppContext";
import { useToast } from "@/hooks/use-toast";
import type { ExtractionResult } from "@shared/schema";

export default function TablesPage() {
  const { selectedExtractionId, setSelectedExtractionId } = useAppContext();
  const [searchQuery, setSearchQuery] = useState("");
  const [sortColumn, setSortColumn] = useState<number | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const { toast } = useToast();

  const { data: extractions, isLoading: extractionsLoading } = useQuery<ExtractionResult[]>({
    queryKey: ["/api/extractions"],
  });

  const { data: selectedData, isLoading: dataLoading } = useQuery<ExtractionResult>({
    queryKey: ["/api/extractions", selectedExtractionId],
    enabled: !!selectedExtractionId,
  });

  const completedExtractions = extractions?.filter(e => e.status === "completed") || [];

  useEffect(() => {
    if (extractionsLoading) return;
    
    const isValidId = selectedExtractionId && completedExtractions.some(e => e.id === selectedExtractionId);
    
    if (!isValidId && completedExtractions.length > 0) {
      setSelectedExtractionId(completedExtractions[0].id);
    } else if (!isValidId && completedExtractions.length === 0) {
      setSelectedExtractionId("");
    }
  }, [extractionsLoading, completedExtractions, selectedExtractionId, setSelectedExtractionId]);

  const handleSort = (columnIndex: number) => {
    if (sortColumn === columnIndex) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(columnIndex);
      setSortDirection("asc");
    }
  };

  let filteredRows = selectedData?.extractedData?.rows?.filter(row => {
    if (!searchQuery) return true;
    return row.some(cell => 
      String(cell).toLowerCase().includes(searchQuery.toLowerCase())
    );
  }) || [];

  if (sortColumn !== null && filteredRows.length > 0) {
    filteredRows = [...filteredRows].sort((a, b) => {
      const aVal = String(a[sortColumn]);
      const bVal = String(b[sortColumn]);
      const comparison = aVal.localeCompare(bVal, undefined, { numeric: true });
      return sortDirection === "asc" ? comparison : -comparison;
    });
  }

  const handleExportCSV = () => {
    if (!selectedData?.extractedData) return;
    
    const csvContent = [
      selectedData.extractedData.headers.join(","),
      ...filteredRows.map(row => row.join(","))
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selectedData.filename}_table.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);

    toast({
      title: "Export Successful",
      description: `Downloaded ${selectedData.filename}_table.csv`,
    });
  };

  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="text-3xl font-semibold mb-2" data-testid="text-tables-title">Extracted Tables</h1>
        <p className="text-muted-foreground">
          View, search, and sort extracted table data from your PDFs
        </p>
      </div>

      <div className="flex flex-wrap gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="text-sm font-medium mb-2 block">Select Extraction</label>
          <Select value={selectedExtractionId} onValueChange={setSelectedExtractionId}>
            <SelectTrigger data-testid="select-extraction">
              <SelectValue placeholder="Choose a completed extraction" />
            </SelectTrigger>
            <SelectContent>
              {extractionsLoading ? (
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

        {selectedExtractionId && (
          <div className="flex-1 min-w-[200px]">
            <label className="text-sm font-medium mb-2 block">Search Table</label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search in table..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
                data-testid="input-search-table"
              />
            </div>
          </div>
        )}
      </div>

      {!selectedExtractionId ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <FileDown className="h-16 w-16 text-muted-foreground/40 mb-4" />
            <p className="text-muted-foreground">Select an extraction to view table data</p>
          </CardContent>
        </Card>
      ) : dataLoading ? (
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-64" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-64 w-full" />
          </CardContent>
        </Card>
      ) : selectedData?.extractedData ? (
        <Card>
          <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-4">
            <div>
              <CardTitle>{selectedData.filename}</CardTitle>
              <CardDescription>
                {selectedData.extractedData.metadata?.totalRows} rows × {selectedData.extractedData.metadata?.totalColumns} columns
                {searchQuery && ` • Showing ${filteredRows.length} of ${selectedData.extractedData.rows.length} rows`}
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={handleExportCSV} data-testid="button-export-csv">
              <Download className="h-4 w-4 mr-2" />
              Export CSV
            </Button>
          </CardHeader>
          <CardContent>
            <div className="border rounded-lg overflow-auto max-h-[600px]">
              <Table>
                <TableHeader className="sticky top-0 bg-card z-10">
                  <TableRow>
                    {selectedData.extractedData.headers.map((header, idx) => (
                      <TableHead 
                        key={idx} 
                        className="font-semibold whitespace-nowrap cursor-pointer hover-elevate"
                        onClick={() => handleSort(idx)}
                        data-testid={`header-${idx}`}
                      >
                        <div className="flex items-center gap-2">
                          {header}
                          <ArrowUpDown className="h-3 w-3" />
                        </div>
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredRows.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={selectedData.extractedData.headers.length} className="text-center py-8 text-muted-foreground">
                        No data found
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredRows.map((row, rowIdx) => (
                      <TableRow 
                        key={rowIdx} 
                        className={rowIdx % 2 === 0 ? "bg-muted/30" : ""}
                        data-testid={`row-${rowIdx}`}
                      >
                        {row.map((cell, cellIdx) => (
                          <TableCell key={cellIdx} className="whitespace-nowrap" data-testid={`cell-${rowIdx}-${cellIdx}`}>
                            {String(cell)}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <p className="text-muted-foreground">No table data available for this extraction</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
