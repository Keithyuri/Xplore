import { FileText, ClipboardList } from "lucide-react";
import type { CaseReport } from "@/types/case";
import CollapsibleSection from "./CollapsibleSection";

interface ReportPanelProps {
  report: CaseReport;
  mode: "raw" | "technical";
  onModeChange: (mode: "raw" | "technical") => void;
}

export default function ReportPanel({ report, mode, onModeChange }: ReportPanelProps) {
  return (
    <div className="flex flex-col h-full bg-card rounded-xl border border-border overflow-hidden">
      {/* Header with mode switcher */}
      <div className="px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2 mb-3">
          <FileText size={16} className="text-primary" />
          <h3 className="text-sm font-semibold text-foreground">PACEMAN Report</h3>
        </div>
        <div className="flex rounded-lg bg-surface p-0.5 border border-border">
          <button
            onClick={() => onModeChange("raw")}
            className={`flex items-center gap-1.5 flex-1 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              mode === "raw"
                ? "bg-primary text-primary-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <FileText size={12} />
            Raw Report
          </button>
          <button
            onClick={() => onModeChange("technical")}
            className={`flex items-center gap-1.5 flex-1 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              mode === "technical"
                ? "bg-primary text-primary-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <ClipboardList size={12} />
            Technical Evaluation
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
        {mode === "raw" ? (
          <div className="prose prose-invert prose-sm max-w-none">
            <pre className="whitespace-pre-wrap text-sm leading-relaxed text-secondary-foreground font-sans">
              {report.rawText}
            </pre>
          </div>
        ) : (
          <div className="space-y-3">
            {report.technicalSections.map((section, i) => (
              <CollapsibleSection
                key={i}
                section={section}
                defaultOpen={
                  i === 0 || section.title.includes("CONCLUSION")
                }
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
