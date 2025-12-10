import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BarChart3, FileText, Upload, Download, Activity, FileSearch } from "lucide-react";
import { Link } from "wouter";

interface FeatureCard {
  title: string;
  description: string;
  icon: React.ElementType;
  path: string;
  gradient: string;
  color: string;
}

const features: FeatureCard[] = [
  {
    title: "Visit Analysis",
    description: "View frequency and distribution",
    icon: BarChart3,
    path: "/analytics",
    gradient: "gradient-primary",
    color: "text-purple-600 dark:text-purple-400"
  },
  {
    title: "Item Tracker",
    description: "Track collection items",
    icon: FileSearch,
    path: "/tables",
    gradient: "gradient-success",
    color: "text-green-600 dark:text-green-400"
  },
  {
    title: "Upload PDF",
    description: "Add new documents",
    icon: Upload,
    path: "/upload",
    gradient: "gradient-info",
    color: "text-blue-600 dark:text-blue-400"
  },
  {
    title: "Annotation Map",
    description: "Reference definitions",
    icon: Activity,
    path: "/analytics",
    gradient: "gradient-warning",
    color: "text-yellow-600 dark:text-yellow-500"
  },
  {
    title: "View Tables",
    description: "Browse extracted data",
    icon: FileText,
    path: "/tables",
    gradient: "gradient-info",
    color: "text-cyan-600 dark:text-cyan-400"
  },
  {
    title: "Export Data",
    description: "Download results",
    icon: Download,
    path: "/export",
    gradient: "gradient-success",
    color: "text-emerald-600 dark:text-emerald-400"
  }
];

export function FeatureCards() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {features.map((feature, index) => {
        const Icon = feature.icon;
        return (
          <Link key={index} href={feature.path}>
            <Card 
              className="feature-card border-2 overflow-hidden cursor-pointer"
              role="button"
              tabIndex={0}
              data-testid={`card-feature-${feature.title.toLowerCase().replace(/\s+/g, '-')}`}
            >
              <CardContent className="p-6">
                <div className="flex items-start gap-4">
                  <div className={`p-3 rounded-lg ${feature.gradient}`}>
                    <Icon className="h-6 w-6 text-white" data-testid={`icon-${feature.title.toLowerCase().replace(/\s+/g, '-')}`} />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-lg mb-1" data-testid={`text-${feature.title.toLowerCase().replace(/\s+/g, '-')}-title`}>
                      {feature.title}
                    </h3>
                    <p className="text-sm text-muted-foreground" data-testid={`text-${feature.title.toLowerCase().replace(/\s+/g, '-')}-description`}>
                      {feature.description}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        );
      })}
    </div>
  );
}
