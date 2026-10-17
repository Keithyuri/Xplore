import type {
  CXRCase,
  TechnicalSection,
  TechnicalSubsection,
} from "@/types/case";

const SIMULATED_DELAY = 400;

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function fetchCases(): Promise<CXRCase[]> {
  await delay(SIMULATED_DELAY);
  return [];
}

export async function fetchCaseById(_caseId: string): Promise<CXRCase | null> {
  await delay(SIMULATED_DELAY);
  return null;
}

export async function fetchOverlayImage(overlayUrl: string): Promise<string> {
  await delay(100);
  return overlayUrl;
}

export async function submitReport(
  _caseId: string,
  _reportData: unknown
): Promise<{ success: boolean }> {
  await delay(200);
  return { success: true };
}

export type SegmentationResult = {
  spinous_mask: string;
  costo_mask: string;
  blunted_mask: string;
  left_clavicle_mask: string;
  right_clavicle_mask: string;
  clavicles_mask: string;
  rotation_overlay: string;
  clavicles_overlay: string;
  spinous_overlay: string;
  costo_overlay: string;
  blunted_overlay: string;
  scapula_overlay: string;
  lung_fields_overlay: string;
  left_scapula_mask: string;
  right_scapula_mask: string;
  left_lung_mask: string;
  right_lung_mask: string;
  centering_overlay: string;
  saved_to?: string;
};

type ParsedReport = {
  mode: "technical";
  rawText: string;
  technicalSections: TechnicalSection[];
};

const emptyReport: ParsedReport = {
  mode: "technical",
  rawText: "No matching report file found for this image.",
  technicalSections: [],
};

function parseTechnicalSections(text: string): TechnicalSection[] {
  if (!text || !text.trim()) return [];

  const normalized = text
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .trim();

  const sectionParts = normalized
    .split(/^#\s+/m)
    .map((part) => part.trim())
    .filter(Boolean);

  return sectionParts.map((sectionPart) => {
    const lines = sectionPart
      .split("\n")
      .map((line) => line.trimEnd());

    const sectionTitle = (lines[0] || "GENERAL").trim();

    const body = lines.slice(1).join("\n").trim();

    const subsectionParts = body
      ? body
          .split(/^##\s+/m)
          .map((part) => part.trim())
          .filter(Boolean)
      : [];

    const subsections: TechnicalSubsection[] = subsectionParts.map((subPart) => {
      const subLines = subPart
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);

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

  const normalized = text
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n");

  const match = normalized.match(
    /=== RAW ===([\s\S]*?)=== TECHNICAL ===([\s\S]*)/
  );

  if (!match) {
    return {
      mode: "technical",
      rawText: normalized.trim(),
      technicalSections: [],
    };
  }

  const rawText = match[1].trim();
  const technicalText = match[2].trim();

  return {
    mode: "technical",
    rawText,
    technicalSections: parseTechnicalSections(technicalText),
  };
}

async function loadReportForFile(fileName: string): Promise<ParsedReport> {
  try {
    const txtFileName = fileName.replace(/\.[^/.]+$/, ".txt");
    const response = await fetch(`/reports/${txtFileName}`);

    if (!response.ok) {
      return emptyReport;
    }

    const text = await response.text();
    const parsed = parseReport(text);

    console.log("Loaded report for:", fileName);
    console.log("Parsed technical sections:", parsed.technicalSections);

    return parsed;
  } catch (error) {
    console.warn("Failed to load report file:", error);
    return emptyReport;
  }
}

export async function segmentXray(file: File): Promise<SegmentationResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://127.0.0.1:8000/segment", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Segmentation failed");
  }

  return response.json();
}

export async function createCaseFromUpload(file: File): Promise<CXRCase> {
  const result = await segmentXray(file);
  const imageUrl = URL.createObjectURL(file);
  const report = await loadReportForFile(file.name);

  return {
    caseId: "uploaded_case",
    caseTitle: `Uploaded Study — ${file.name}`,
    description:
      "Uploaded chest radiograph processed by the backend segmentation pipeline.",
    imageUrl,
    overlays: [
      {
        id: "rotation",
        name: "Rotation",
        visible: false,
        opacity: 1,
        color: "#f59e0b",
        type: "overlay",
        dataUrl: `data:image/png;base64,${result.rotation_overlay}`,
      },
      {
        id: "clavicles",
        name: "Clavicles",
        visible: false,
        opacity: 1,
        color: "#5900ff",
        type: "overlay",
        dataUrl: `data:image/png;base64,${result.clavicles_overlay}`,
      },
      {
        id: "spinous_process",
        name: "Spinous Process",
        visible: false,
        opacity: 1,
        color: "#00FFFF",
        type: "overlay",
        dataUrl: `data:image/png;base64,${result.spinous_overlay}`,
      },
      {
        id: "costophrenic_angle",
        name: "Costophrenic Angle",
        visible: false,
        opacity: 1,
        color: "#3b82f6",
        type: "overlay",
        dataUrl: `data:image/png;base64,${result.costo_overlay}`,
      },
      {
        id: "scapula",
        name: "Scapula",
        visible: false,
        opacity: 1,
        color: "#de16f9",
        type: "overlay",
        dataUrl: `data:image/png;base64,${result.scapula_overlay}`,
      },
      {
        id: "lung_fields",
        name: "Overlapping Lung Fields",
        visible: false,
        opacity: 1,
        color: "#ffe600",
        type: "overlay",
        dataUrl: `data:image/png;base64,${result.lung_fields_overlay}`,
      },
      {
        id: "centering",
        name: "Centering",
        visible: false,
        opacity: 1,
        color: "#22c55e",
        type: "overlay",
        dataUrl: `data:image/png;base64,${result.centering_overlay}`,
      },
    ],
    report,
  };
}