import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FileText, BarChart3, Download, Zap } from "lucide-react";
import { FeatureCards } from "@/components/feature-cards";

export default function Landing() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center justify-center px-6 py-24">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight leading-tight">
            Advanced Clinical PDF
            <br />
            <span className="text-primary">Data Extraction</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Extract, analyze, and visualize clinical trial data from PDF documents with
            AI-powered OCR and advanced analytics. Built for precision and efficiency.
          </p>
          <div className="flex gap-4 justify-center pt-4">
            <Link href="/upload">
              <Button size="lg" className="h-12 px-8" data-testid="button-get-started">
                Get Started
              </Button>
            </Link>
            <Link href="/dashboard">
              <Button size="lg" variant="outline" className="h-12 px-8" data-testid="button-view-dashboard">
                View Dashboard
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Quick Access Feature Cards */}
      <section className="py-16 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12 space-y-4">
            <h2 className="text-3xl md:text-4xl font-bold">Quick Access</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Navigate to key features with one click
            </p>
          </div>
          <FeatureCards />
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 px-6 bg-card/50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 space-y-4">
            <h2 className="text-3xl md:text-4xl font-bold">Powerful Features</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Everything you need to extract and analyze clinical trial data efficiently
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="hover-elevate">
              <CardHeader>
                <FileText className="h-10 w-10 text-primary mb-2" />
                <CardTitle>Smart Extraction</CardTitle>
                <CardDescription>
                  Automatic detection of vector and scanned PDFs with dual extraction methods
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="hover-elevate">
              <CardHeader>
                <Zap className="h-10 w-10 text-primary mb-2" />
                <CardTitle>AI-Powered OCR</CardTitle>
                <CardDescription>
                  Mistral OCR via Groq for complex tables with multi-page stitching
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="hover-elevate">
              <CardHeader>
                <BarChart3 className="h-10 w-10 text-primary mb-2" />
                <CardTitle>Advanced Analytics</CardTitle>
                <CardDescription>
                  Interactive charts for visit frequency, period analysis, and assessments
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="hover-elevate">
              <CardHeader>
                <Download className="h-10 w-10 text-primary mb-2" />
                <CardTitle>Multi-Format Export</CardTitle>
                <CardDescription>
                  Download extracted data in CSV, Excel, or JSON formats
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto text-center space-y-6 bg-primary text-primary-foreground rounded-lg p-12">
          <h2 className="text-3xl md:text-4xl font-bold">
            Ready to Extract Clinical Data?
          </h2>
          <p className="text-lg opacity-90">
            Upload your first PDF and see the power of advanced extraction
          </p>
          <Link href="/upload">
            <Button size="lg" variant="secondary" className="h-12 px-8" data-testid="button-upload-now">
              Upload Now
            </Button>
          </Link>
        </div>
      </section>
    </div>
  );
}
