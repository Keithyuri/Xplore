export interface Overlay {
  id: string;
  name: string;
  visible: boolean;
  type: "mask" | "contour" | "annotation";
  opacity: number;
  color: string;
  dataUrl: string;
}

export interface ReportSubsection {
  title: string;
  content: string[];
  severity?: "mild" | "moderate" | "severe";
}

export interface ReportSection {
  title: string;
  subsections: ReportSubsection[];
}

export interface CaseReport {
  mode: "raw" | "technical";
  rawText: string;
  technicalSections: ReportSection[];
}

export interface CXRCase {
  caseId: string;
  caseTitle: string;
  description: string;
  imageUrl: string;
  overlays: Overlay[];
  report: CaseReport;
}

export type AppState = "loading" | "ready" | "error" | "empty";
