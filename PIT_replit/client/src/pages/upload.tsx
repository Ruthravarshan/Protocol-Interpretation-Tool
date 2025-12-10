import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Upload, FileText, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import type { UploadResponse } from "@shared/schema";

export default function UploadPage() {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: async (file: File): Promise<UploadResponse> => {
      const formData = new FormData();
      formData.append("file", file);
      
      const response = await apiRequest("POST", "/api/upload", formData);
      return response.json();
    },
    onSuccess: (data: UploadResponse) => {
      toast({
        title: "Upload Successful",
        description: `${data.filename} has been uploaded and is being processed.`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/dashboard"] });
      queryClient.invalidateQueries({ queryKey: ["/api/extractions"] });
      setUploadProgress(100);
      
      // Start polling for extraction status
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: [`/api/extractions/${data.id}`] });
      }, 2000);
    },
    onError: (error: Error) => {
      toast({
        title: "Upload Failed",
        description: error.message || "Failed to upload file. Please try again.",
        variant: "destructive",
      });
      setUploadProgress(0);
      setUploadedFile(null);
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setUploadedFile(file);
      setUploadProgress(0);
      uploadMutation.mutate(file);
    }
  }, [uploadMutation]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
    },
    maxFiles: 1,
    multiple: false,
  });

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-semibold mb-2" data-testid="text-upload-title">Upload PDF</h1>
        <p className="text-muted-foreground">
          Upload clinical trial PDF documents for extraction and analysis
        </p>
      </div>

      {/* Dropzone */}
      <Card>
        <CardHeader>
          <CardTitle>Upload Document</CardTitle>
          <CardDescription>
            Drag and drop your PDF file here, or click to select
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={`min-h-64 border-2 border-dashed rounded-lg flex flex-col items-center justify-center p-12 cursor-pointer transition-colors ${
              isDragActive
                ? "border-primary bg-primary/10"
                : "border-border hover:border-primary hover:bg-primary/5"
            }`}
            data-testid="dropzone-upload"
          >
            <input {...getInputProps()} />
            <Upload className="h-12 w-12 text-muted-foreground mb-4" />
            {isDragActive ? (
              <p className="text-lg font-medium">Drop the PDF file here</p>
            ) : (
              <>
                <p className="text-lg font-medium mb-2">
                  Drop PDF file here or click to browse
                </p>
                <p className="text-sm text-muted-foreground">
                  Supports PDF files up to 50 pages
                </p>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Upload Status */}
      {uploadedFile && (
        <Card>
          <CardHeader>
            <CardTitle>Processing Status</CardTitle>
            <CardDescription>Current file being processed</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              <FileText className="h-8 w-8 text-primary" />
              <div className="flex-1">
                <p className="font-medium">{uploadedFile.name}</p>
                <p className="text-sm text-muted-foreground">
                  {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              {uploadMutation.isPending && (
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
              )}
              {uploadMutation.isSuccess && (
                <CheckCircle className="h-5 w-5 text-green-600" />
              )}
              {uploadMutation.isError && (
                <XCircle className="h-5 w-5 text-destructive" />
              )}
            </div>

            {uploadMutation.isPending && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Uploading...</span>
                  <span>{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} className="h-2" />
              </div>
            )}

            {uploadMutation.isSuccess && (
              <div className="p-4 bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-lg">
                <p className="text-sm text-green-800 dark:text-green-200">
                  File uploaded successfully! Processing extraction...
                </p>
              </div>
            )}

            {uploadMutation.isError && (
              <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                <p className="text-sm text-destructive">
                  Upload failed. Please try again.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Instructions */}
      <Card>
        <CardHeader>
          <CardTitle>Extraction Process</CardTitle>
          <CardDescription>How the system processes your documents</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
              1
            </div>
            <div>
              <h4 className="font-medium mb-1">Auto-Detection</h4>
              <p className="text-sm text-muted-foreground">
                System automatically detects if your PDF is vector-based or scanned
              </p>
            </div>
          </div>

          <div className="flex gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
              2
            </div>
            <div>
              <h4 className="font-medium mb-1">Smart Extraction</h4>
              <p className="text-sm text-muted-foreground">
                Uses Camelot for vector PDFs or Mistral OCR for scanned documents
              </p>
            </div>
          </div>

          <div className="flex gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
              3
            </div>
            <div>
              <h4 className="font-medium mb-1">Data Processing</h4>
              <p className="text-sm text-muted-foreground">
                Multi-page stitching, header normalization, and analytics generation
              </p>
            </div>
          </div>

          <div className="flex gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
              4
            </div>
            <div>
              <h4 className="font-medium mb-1">Results Available</h4>
              <p className="text-sm text-muted-foreground">
                View tables, analytics, and export data in multiple formats
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
