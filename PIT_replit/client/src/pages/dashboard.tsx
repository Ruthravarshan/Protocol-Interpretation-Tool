import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { FileText, CheckCircle, XCircle, Loader2, Users, Calendar, Database, BookOpen, FlaskConical, MessageSquare, Beaker, ChevronDown, Activity, BarChart } from "lucide-react";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useAppContext } from "@/contexts/AppContext";
import type { DashboardStats, ExtractionResult, ClinicalMetrics } from "@shared/schema";

const formatVisitName = (visit: string): string => {
  const visitMatch = visit.match(/^Visit\s+([\d.]+)/i);
  if (visitMatch) {
    return `V ${visitMatch[1]}`;
  }
  const vMatch = visit.match(/^V\s*([\d.]+)\s*\(/i);
  if (vMatch) {
    return `V ${vMatch[1]}`;
  }
  const simpleVMatch = visit.match(/^V\s*([\d.]+)$/i);
  if (simpleVMatch) {
    return `V ${simpleVMatch[1]}`;
  }
  return visit;
};

export default function Dashboard() {
  const [analyticsOpen, setAnalyticsOpen] = useState(true);
  const { selectedExtractionId, setSelectedExtractionId } = useAppContext();

  const { data: stats, isLoading } = useQuery<DashboardStats>({
    queryKey: ["/api/dashboard"],
  });

  const { data: clinicalMetrics, isLoading: clinicalMetricsLoading } = useQuery<ClinicalMetrics>({
    queryKey: ["/api/clinical-metrics"],
  });

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

  if (isLoading) {
    return (
      <div className="p-8 space-y-8">
        <div>
          <h1 className="text-3xl font-semibold mb-2">Dashboard</h1>
          <p className="text-muted-foreground">Overview of your PDF extractions</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardHeader className="space-y-0 pb-2">
                <Skeleton className="h-4 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-16" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  const kpiCards = [
    {
      title: "Total Extractions",
      value: stats?.totalExtractions || 0,
      icon: FileText,
      color: "text-primary",
    },
    {
      title: "Successful",
      value: stats?.successfulExtractions || 0,
      icon: CheckCircle,
      color: "text-green-600",
    },
    {
      title: "Failed",
      value: stats?.failedExtractions || 0,
      icon: XCircle,
      color: "text-destructive",
    },
    {
      title: "Processing",
      value: stats?.processingExtractions || 0,
      icon: Loader2,
      color: "text-blue-600",
    },
  ];

  return (
    <div className="p-8 space-y-8">
      <div className="flex justify-between items-center flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-semibold mb-2" data-testid="text-dashboard-title">Dashboard</h1>
          <p className="text-muted-foreground">Overview of your PDF extractions</p>
        </div>
        <Link href="/upload">
          <Button data-testid="button-new-extraction">New Extraction</Button>
        </Link>
      </div>

      <div className="max-w-md">
        <label className="text-sm font-medium mb-2 block" data-testid="label-select-extracted-pdf">Select Extracted PDF</label>
        <Select value={selectedExtractionId} onValueChange={setSelectedExtractionId}>
          <SelectTrigger data-testid="select-extracted-pdf">
            <SelectValue placeholder="Choose an extracted PDF" />
          </SelectTrigger>
          <SelectContent>
            {extractionsLoading ? (
              <SelectItem value="loading" disabled data-testid="select-item-loading">Loading...</SelectItem>
            ) : completedExtractions.length === 0 ? (
              <SelectItem value="none" disabled data-testid="select-item-empty">No completed extractions available</SelectItem>
            ) : (
              completedExtractions.map(extraction => (
                <SelectItem key={extraction.id} value={extraction.id} data-testid={`select-item-pdf-${extraction.id}`}>
                  {extraction.filename}
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {kpiCards.map((kpi) => (
          <Card key={kpi.title} data-testid={`card-${kpi.title.toLowerCase().replace(' ', '-')}`}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 gap-2">
              <CardTitle className="text-sm font-medium">{kpi.title}</CardTitle>
              <kpi.icon className={`h-4 w-4 ${kpi.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{kpi.value}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      {stats && stats.totalExtractions > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Success Rate</CardTitle>
            <CardDescription>Percentage of successful extractions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Success Rate</span>
                <span className="font-semibold">{stats.successRate.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-3">
                <div
                  className="bg-primary h-3 rounded-full transition-all"
                  style={{ width: `${stats.successRate}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-semibold mb-2" data-testid="text-clinical-metrics-title">Clinical Study Metrics</h2>
          <p className="text-muted-foreground">Overview of study data and collections</p>
        </div>

        {clinicalMetricsLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[...Array(7)].map((_, i) => (
              <Card key={i}>
                <CardHeader className="space-y-0 pb-2">
                  <Skeleton className="h-4 w-24" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-8 w-16" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card data-testid="card-total-subjects">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 gap-2">
                <CardTitle className="text-sm font-medium">Total Subjects</CardTitle>
                <Users className="h-4 w-4 text-blue-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold" data-testid="text-total-subjects">
                  {clinicalMetrics?.totalSubjects || 0}
                </div>
              </CardContent>
            </Card>

            <Card data-testid="card-total-visits">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 gap-2">
                <CardTitle className="text-sm font-medium">Total Visits</CardTitle>
                <Calendar className="h-4 w-4 text-green-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold" data-testid="text-total-visits">
                  {clinicalMetrics?.totalVisits || 0}
                </div>
              </CardContent>
            </Card>

            <Card data-testid="card-unique-items-per-study">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 gap-2">
                <CardTitle className="text-sm font-medium">Unique Items per Study</CardTitle>
                <Beaker className="h-4 w-4 text-indigo-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold" data-testid="text-unique-items-per-study">
                  29
                </div>
              </CardContent>
            </Card>

            <Card data-testid="card-total-unique-items">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 gap-2">
                <CardTitle className="text-sm font-medium">Total Unique Items</CardTitle>
                <Database className="h-4 w-4 text-purple-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold" data-testid="text-total-unique-items">
                  {clinicalMetrics ? (clinicalMetrics.totalItems * clinicalMetrics.totalSubjects) : 0}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {clinicalMetrics && `${clinicalMetrics.totalItems} items x ${clinicalMetrics.totalSubjects} subjects`}
                </p>
              </CardContent>
            </Card>

            <Card data-testid="card-total-collections">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 gap-2">
                <CardTitle className="text-sm font-medium">Total Collections</CardTitle>
                <BookOpen className="h-4 w-4 text-orange-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold" data-testid="text-total-collections">
                  {clinicalMetrics?.totalCollections || 0}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {clinicalMetrics && `${clinicalMetrics.totalSubjects} x ${clinicalMetrics.collectionsPreStudy} collections per study`}
                </p>
              </CardContent>
            </Card>

            <Card data-testid="card-collections-perstudy">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 gap-2">
                <CardTitle className="text-sm font-medium">Collections Per Study</CardTitle>
                <FlaskConical className="h-4 w-4 text-pink-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold" data-testid="text-collections-perstudy">
                  {clinicalMetrics?.collectionsPreStudy || 0}
                </div>
              </CardContent>
            </Card>

            <Card data-testid="card-study-periods">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 gap-2">
                <CardTitle className="text-sm font-medium">Study Periods</CardTitle>
                <Calendar className="h-4 w-4 text-teal-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold" data-testid="text-study-periods">
                  {clinicalMetrics?.studyPeriods || 0}
                </div>
              </CardContent>
            </Card>

            <Card className="bg-primary/10 border-primary" data-testid="card-annotations">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 gap-2">
                <CardTitle className="text-sm font-medium">Annotations</CardTitle>
                <MessageSquare className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-primary" data-testid="text-annotations">
                  {clinicalMetrics?.annotations || 0}
                </div>
                <p className="text-xs text-muted-foreground mt-1">Total annotations in study</p>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      <Collapsible open={analyticsOpen} onOpenChange={setAnalyticsOpen}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover-elevate" data-testid="button-analytics-toggle">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <BarChart className="h-5 w-5 text-primary" />
                  <CardTitle>Analytics</CardTitle>
                </div>
                <ChevronDown className={`h-5 w-5 transition-transform ${analyticsOpen ? 'rotate-180' : ''}`} />
              </div>
              <CardDescription>Advanced visualizations and insights from clinical trial visit data</CardDescription>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent className="space-y-6">
              {!selectedExtractionId ? (
                <div className="flex flex-col items-center justify-center py-12">
                  <Activity className="h-16 w-16 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground text-lg">Select an extraction above to view analytics</p>
                </div>
              ) : dataLoading ? (
                <div className="grid lg:grid-cols-2 gap-6">
                  {[...Array(4)].map((_, i) => (
                    <div key={i} className="border-2 rounded-lg p-6">
                      <Skeleton className="h-6 w-48 mb-4" />
                      <Skeleton className="h-64 w-full" />
                    </div>
                  ))}
                </div>
              ) : analytics ? (
                <div className="space-y-6">
                  {analytics.visitFrequency && analytics.visitFrequency.length > 0 && (
                    <div className="border-2 rounded-lg p-6">
                      <div className="flex items-center gap-2 mb-2">
                        <BarChart className="h-5 w-5 text-primary" />
                        <h3 className="font-semibold">Visit Analysis</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-4">Distribution of items across visits</p>
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
                    </div>
                  )}

                  {analytics.assessmentsByVisit && analytics.assessmentsByVisit.length > 0 && (
                    <div className="border-2 rounded-lg p-6">
                      <div className="flex items-center gap-2 mb-2">
                        <FileText className="h-5 w-5 text-accent" />
                        <h3 className="font-semibold">Collection Items Analysis</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-4">Items collected and their visit locations</p>
                      <div className="overflow-auto">
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
                      </div>
                    </div>
                  )}

                  {analytics.periodAnalysis && analytics.periodAnalysis.length > 0 && (
                    <div className="border-2 rounded-lg p-6">
                      <div className="flex items-center gap-2 mb-2">
                        <Activity className="h-5 w-5 text-secondary" />
                        <h3 className="font-semibold">Period Analysis</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-4">Visits grouped by study period</p>
                      <div className="overflow-auto">
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
                      </div>
                    </div>
                  )}

                  {analytics.assessmentsByVisit && analytics.assessmentsByVisit.length > 0 && (
                    <div className="border-2 rounded-lg p-6">
                      <div className="flex items-center gap-2 mb-2">
                        <FileText className="h-5 w-5 text-info" />
                        <h3 className="font-semibold">Detailed Assessments by Visit</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-4">Complete breakdown of procedures per visit</p>
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
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12">
                  <Activity className="h-16 w-16 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground text-lg">No analytics data available for this extraction</p>
                </div>
              )}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      <Card>
        <CardHeader>
          <CardTitle>Recent Extractions</CardTitle>
          <CardDescription>Your latest PDF processing activity</CardDescription>
        </CardHeader>
        <CardContent>
          {!stats?.recentExtractions || stats.recentExtractions.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No extractions yet</p>
              <p className="text-sm">Upload a PDF to get started</p>
            </div>
          ) : (
            <div className="space-y-4">
              {stats.recentExtractions.map((extraction: ExtractionResult) => (
                <div
                  key={extraction.id}
                  className="flex items-center justify-between p-4 rounded-lg border hover-elevate"
                  data-testid={`extraction-${extraction.id}`}
                >
                  <div className="flex-1">
                    <h4 className="font-medium">{extraction.filename}</h4>
                    <p className="text-sm text-muted-foreground">
                      {new Date(extraction.uploadedAt).toLocaleString()}
                    </p>
                  </div>
                  <Badge
                    variant={
                      extraction.status === "completed"
                        ? "default"
                        : extraction.status === "failed"
                        ? "destructive"
                        : "secondary"
                    }
                    data-testid={`badge-status-${extraction.id}`}
                  >
                    {extraction.status}
                  </Badge>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
