import { useState } from "react";
import CaseViewer from "./CaseViewer";
import type { CXRCase } from "@/types/case";
import { createCaseFromUpload } from "@/services/api";

function parseTechnicalSections(text: string): TechnicalSection[] {
  if (!text || !text.trim()) return [];

  const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();

  const sectionParts = normalized
    .split(/^#\s+/m)
    .map((part) => part.trim())
    .filter(Boolean);

  return sectionParts.map((sectionPart) => {
    const lines = sectionPart.split("\n");
    const sectionTitle = lines[0].trim();

    const body = lines.slice(1).join("\n").trim();

    const subsectionParts = body
      .split(/^##\s+/m)
      .map((part) => part.trim())
      .filter(Boolean);

    const subsections: TechnicalSubsection[] = subsectionParts.map((subPart) => {
      const subLines = subPart.split("\n").map((line) => line.trim()).filter(Boolean);
      const subsectionTitle = subLines[0] || "Details";
      const content = subLines.slice(1);

      return {
        title: subsectionTitle,
        content: content.length ? content : [""],
      };
    });

    return {
      title: sectionTitle,
      subsections,
    };
  });
}

function parseReport(text: string): ParsedReport {
  if (!text || !text.trim()) return emptyReport;

  const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

  const rawMatch = normalized.match(
    /=== RAW ===([\s\S]*?)=== TECHNICAL ===([\s\S]*)/
  );

  if (!rawMatch) {
    return {
      mode: "technical",
      rawText: normalized.trim(),
      technicalSections: [],
    };
  }

  const rawText = rawMatch[1].trim();
  const technicalText = rawMatch[2].trim();

  return {
    mode: "technical",
    rawText,
    technicalSections: parseTechnicalSections(technicalText),
  };
}

const Index = () => {
  const [uploadedCase, setUploadedCase] = useState<CXRCase | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async (file: File) => {
    try {
      setLoading(true);
      setError("");

      const newCase = await createCaseFromUpload(file);

      let reportData = null;

      try {
        const txtFileName = file.name.replace(/\.[^/.]+$/, ".txt");
        const response = await fetch(`/reports/${txtFileName}`);

        if (response.ok) {
          const text = await response.text();
          reportData = parseReport(text);
        }
      } catch (reportErr) {
        console.warn("Could not load matching report file:", reportErr);
      }

      const updatedCase = {
        ...newCase,
        report: reportData
          ? {
              mode: "technical",
              rawText: reportData.rawText,
              technicalSections: reportData.technicalSections,
            }
          : newCase.report,
      };

      setUploadedCase(updatedCase);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen overflow-hidden bg-background">
      <CaseViewer
        caseId={uploadedCase?.caseId}
        uploadedCase={uploadedCase}
        onUploadImage={handleUpload}
        isUploading={loading}
      />

      {error && (
        <div className="absolute left-4 top-4 z-50 rounded-md border border-destructive/30 bg-card px-3 py-2 text-sm text-destructive shadow-lg">
          {error}
        </div>
      )}
    </div>
  );
};

export default Index;