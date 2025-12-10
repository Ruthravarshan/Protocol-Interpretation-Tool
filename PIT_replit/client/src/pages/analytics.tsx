import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { useEffect } from "react";
import { Activity, FileText, BarChart } from "lucide-react";
import { useAppContext } from "@/contexts/AppContext";
import type { ExtractionResult } from "@shared/schema";

// Helper function to format visit names:
// "Visit 0 Stabilization" → "V 0", "Visit 1 Screening" → "V 1"
// "Visit 1.1 Baseline" → "V 1.1", "Visit 2 Baselin" → "V 2"
// "V 8 (EoT Visit)" → "V 8", "V 9 (Safety Follow-up Visit)" → "V 9"
const formatVisitName = (visit: string): string => {
  // Handle "Visit X Description" format (e.g., "Visit 0 Stabilization", "Visit 1.1 Baseline")
  const visitMatch = visit.match(/^Visit\s+([\d.]+)/i);
  if (visitMatch) {
    return `V ${visitMatch[1]}`;
  }
  
  // Handle "V X (description)" format (e.g., "V 8 (EoT Visit)")
  const vMatch = visit.match(/^V\s*([\d.]+)\s*\(/i);
  if (vMatch) {
    return `V ${vMatch[1]}`;
  }
  
  // Handle simple "V X" format that's already correct
  const simpleVMatch = visit.match(/^V\s*([\d.]+)$/i);
  if (simpleVMatch) {
    return `V ${simpleVMatch[1]}`;
  }
  
  return visit;
};

const COLORS = [
  'hsl(var(--chart-1))',
  'hsl(var(--chart-2))',
  'hsl(var(--chart-3))',
  'hsl(var(--chart-4))',
  'hsl(var(--chart-5))'
];

export default function AnalyticsPage() {
  const { selectedExtractionId, setSelectedExtractionId } = useAppContext();

  const { data: extractions, isLoading: extractionsLoading } = useQuery<ExtractionResult[]>({
    queryKey: ["/api/extractions"],
  });

  const { data: selectedData, isLoading: dataLoading } = useQuery<ExtractionResult>({
    queryKey: ["/api/extractions", selectedExtractionId],
    enabled: !!selectedExtractionId,
  });

  const completedExtractions = extractions?.filter(e => e.status === "completed" && e.analyticsData) || [];
  const analytics = selectedData?.analyticsData;

  useEffect(() => {
    if (extractionsLoading) return;
    
    const isValidId = selectedExtractionId && completedExtractions.some(e => e.id === selectedExtractionId);
    
    if (!isValidId && completedExtractions.length > 0) {
      setSelectedExtractionId(completedExtractions[0].id);
    } else if (!isValidId && completedExtractions.length === 0) {
      setSelectedExtractionId("");
    }
  }, [extractionsLoading, completedExtractions, selectedExtractionId, setSelectedExtractionId]);

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-semibold mb-2" data-testid="text-visit-analysis-title">Visit Analysis</h1>
        <p className="text-muted-foreground">
          Advanced visualizations and insights from clinical trial visit data
        </p>
      </div>

      {/* Extraction Selector */}
      <div className="max-w-md">
        <label className="text-sm font-medium mb-2 block">Select Extraction</label>
        <Select value={selectedExtractionId} onValueChange={setSelectedExtractionId}>
          <SelectTrigger data-testid="select-extraction-analytics">
            <SelectValue placeholder="Choose a completed extraction" />
          </SelectTrigger>
          <SelectContent>
            {extractionsLoading ? (
              <SelectItem value="loading" disabled>Loading...</SelectItem>
            ) : completedExtractions.length === 0 ? (
              <SelectItem value="none" disabled>No completed extractions with analytics</SelectItem>
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

      {!selectedExtractionId ? (
        <Card className="border-2">
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Activity className="h-16 w-16 text-muted-foreground mb-4" />
            <p className="text-muted-foreground text-lg">Select an extraction to view analytics</p>
          </CardContent>
        </Card>
      ) : dataLoading ? (
        <div className="grid lg:grid-cols-2 gap-6">
          {[...Array(6)].map((_, i) => (
            <Card key={i} className="border-2">
              <CardHeader>
                <Skeleton className="h-6 w-48" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-64 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : analytics ? (
        <>
          {/* Visit Analysis Chart */}
          {analytics.visitFrequency && analytics.visitFrequency.length > 0 && (
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart className="h-5 w-5 text-primary" />
                  Visit Analysis
                </CardTitle>
                <CardDescription>Distribution of items across visits</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <RechartsBarChart data={analytics.visitFrequency.map(v => ({ ...v, visit: formatVisitName(v.visit) }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis 
                      dataKey="visit" 
                      angle={-45} 
                      textAnchor="end" 
                      height={100}
                      label={{ value: 'Visit', position: 'insideBottom', offset: -5 }}
                      stroke="hsl(var(--foreground))"
                    />
                    <YAxis 
                      label={{ value: 'Number of Items', angle: -90, position: 'insideLeft' }}
                      stroke="hsl(var(--foreground))"
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'hsl(var(--card))', 
                        border: '2px solid hsl(var(--border))',
                        borderRadius: '0.5rem'
                      }}
                    />
                    <Legend />
                    <Bar 
                      dataKey="count" 
                      fill="hsl(var(--primary))" 
                      name="Number of Items"
                      radius={[8, 8, 0, 0]}
                    />
                  </RechartsBarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Collection Items Analysis */}
          {analytics.assessmentsByVisit && analytics.assessmentsByVisit.length > 0 && (
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-accent" />
                  Collection Items Analysis
                </CardTitle>
                <CardDescription>Items collected and their visit locations</CardDescription>
              </CardHeader>
              <CardContent className="overflow-auto">
                <table className="enhanced-table w-full">
                  <thead>
                    <tr>
                      <th>Item Name</th>
                      <th className="text-center">Collection Count</th>
                      <th>Visits Where Collected</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analytics.assessmentsByVisit.flatMap(visit => 
                      visit.assessments.slice(0, 3).map((assessment, idx) => ({
                        name: assessment,
                        visit: visit.visit,
                        key: `${visit.visit}-${idx}`
                      }))
                    ).reduce((acc, curr) => {
                      const existing = acc.find(item => item.name === curr.name);
                      if (existing) {
                        existing.count++;
                        if (!existing.visits.includes(curr.visit)) {
                          existing.visits.push(curr.visit);
                        }
                      } else {
                        acc.push({ name: curr.name, count: 1, visits: [curr.visit] });
                      }
                      return acc;
                    }, [] as Array<{name: string, count: number, visits: string[]}>)
                    .slice(0, 10)
                    .map((item, idx) => (
                      <tr key={idx}>
                        <td className="font-medium">{item.name}</td>
                        <td className="text-center">
                          <Badge variant="secondary" className="font-bold">
                            {item.count}
                          </Badge>
                        </td>
                        <td>
                          <div className="flex flex-wrap gap-1">
                            {item.visits.map((visit, vIdx) => (
                              <Badge key={vIdx} className="text-xs">
                                {formatVisitName(visit)}
                              </Badge>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}

          {/* Period Analysis */}
          {analytics.periodAnalysis && analytics.periodAnalysis.length > 0 && (
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-secondary" />
                  Period Analysis
                </CardTitle>
                <CardDescription>Visits grouped by study period</CardDescription>
              </CardHeader>
              <CardContent className="overflow-auto">
                <table className="enhanced-table w-full">
                  <thead>
                    <tr>
                      <th>Period</th>
                      <th>Visits</th>
                      <th className="text-right">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analytics.periodAnalysis.map((period, idx) => (
                      <tr key={idx}>
                        <td className="font-semibold">{period.period}</td>
                        <td>
                          <div className="flex flex-wrap gap-1">
                            {period.visits.map((visit, vIdx) => (
                              <Badge key={vIdx} variant="outline" className="text-xs">
                                {formatVisitName(visit)}
                              </Badge>
                            ))}
                          </div>
                        </td>
                        <td className="text-right font-bold">{period.visits.length}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}

          {/* Assessments by Visit Details */}
          {analytics.assessmentsByVisit && analytics.assessmentsByVisit.length > 0 && (
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-info" />
                  Detailed Assessments by Visit
                </CardTitle>
                <CardDescription>Complete breakdown of procedures per visit</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {analytics.assessmentsByVisit.map((visit, idx) => (
                    <div key={idx} className="p-4 border-2 rounded-lg hover-elevate">
                      <div className="flex justify-between items-start mb-3">
                        <h4 className="font-bold text-lg text-primary">{formatVisitName(visit.visit)}</h4>
                        <Badge className="text-sm">
                          {visit.count} assessments
                        </Badge>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {visit.assessments.slice(0, 15).map((assessment, aIdx) => (
                          <Badge key={aIdx} variant="secondary" className="text-xs px-3 py-1">
                            {assessment}
                          </Badge>
                        ))}
                        {visit.assessments.length > 15 && (
                          <Badge variant="outline" className="text-xs px-3 py-1">
                            +{visit.assessments.length - 15} more
                          </Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      ) : (
        <Card className="border-2">
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Activity className="h-16 w-16 text-muted-foreground mb-4" />
            <p className="text-muted-foreground text-lg">No analytics data available for this extraction</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
