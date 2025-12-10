import type { Express } from "express";
import { createServer, type Server } from "http";
import { spawn } from "child_process";
import multer from "multer";
import fetch from "node-fetch";
import FormData from "form-data";
import fs from "fs";

const upload = multer({ storage: multer.memoryStorage() });

let pythonProcess: any = null;

// Start Python FastAPI server
async function startPythonServer() {
  if (pythonProcess) {
    return;
  }

  console.log("Starting Python FastAPI server on port 8000...");
  
  pythonProcess = spawn("python3", ["server/python/main.py"], {
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
    },
  });

  if (pythonProcess.stdout) {
    pythonProcess.stdout.on("data", (data: Buffer) => {
      console.log(`[Python] ${data.toString().trim()}`);
    });
  }

  if (pythonProcess.stderr) {
    pythonProcess.stderr.on("data", (data: Buffer) => {
      console.error(`[Python Error] ${data.toString().trim()}`);
    });
  }

  pythonProcess.on("error", (error: Error) => {
    console.error("Failed to start Python server:", error);
  });

  pythonProcess.on("exit", (code: number) => {
    console.log(`Python server exited with code ${code}`);
    pythonProcess = null;
  });

  // Give the server time to start
  await new Promise((resolve) => setTimeout(resolve, 3000));
  console.log("Python server should be ready");
}

export async function registerRoutes(app: Express): Promise<Server> {
  // Start Python backend
  await startPythonServer();

  // Handle file upload specifically
  app.post("/api/upload", upload.single("file"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      const formData = new FormData();
      formData.append("file", req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype,
      });

      const response = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData as any,
        headers: formData.getHeaders(),
      });

      if (!response.ok) {
        throw new Error(`Python API responded with ${response.status}`);
      }

      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error("Upload proxy error:", error);
      res.status(500).json({
        error: "Failed to process upload",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Proxy all other /api requests to Python FastAPI backend
  app.use("/api/*", async (req, res) => {
    try {
      const pythonUrl = `http://localhost:8000${req.originalUrl}`;
      
      const options: any = {
        method: req.method,
        headers: {
          "content-type": "application/json",
        },
      };

      if (req.method !== "GET" && req.method !== "HEAD" && req.body) {
        options.body = JSON.stringify(req.body);
      }

      const response = await fetch(pythonUrl, options);
      
      // Copy headers from Python response
      const contentType = response.headers.get("content-type") || "";
      response.headers.forEach((value, key) => {
        res.setHeader(key, value);
      });

      res.status(response.status);

      // Handle different response types
      if (contentType.includes("application/json")) {
        const data = await response.json();
        res.json(data);
      } else if (contentType.includes("text/")) {
        const text = await response.text();
        res.send(text);
      } else {
        // Binary data (Excel, CSV, etc.)
        const buffer = await response.arrayBuffer();
        res.send(Buffer.from(buffer));
      }
    } catch (error) {
      console.error("API proxy error:", error);
      res.status(500).json({
        error: "Failed to communicate with Python backend",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  });

  const httpServer = createServer(app);

  // Cleanup on shutdown
  process.on("SIGTERM", () => {
    if (pythonProcess) {
      pythonProcess.kill();
    }
  });

  process.on("SIGINT", () => {
    if (pythonProcess) {
      pythonProcess.kill();
    }
    process.exit();
  });

  return httpServer;
}
