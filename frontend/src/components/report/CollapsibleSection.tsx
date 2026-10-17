import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { ReportSection } from "@/types/case";
import SeverityBadge from "./SeverityBadge";

interface CollapsibleSectionProps {
  section: ReportSection;
  defaultOpen?: boolean;
}

export default function CollapsibleSection({ section, defaultOpen = false }: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const isFinal = section.title.includes("CONCLUSION");

  return (
    <div
      className={`border border-border rounded-lg overflow-hidden transition-colors ${
        isFinal ? "border-primary/40 glow-primary" : ""
      }`}
    >
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center justify-between w-full px-4 py-3 text-left transition-colors ${
          isFinal
            ? "bg-primary/10 hover:bg-primary/15"
            : "bg-card hover:bg-surface-hover"
        }`}
      >
        <h4
          className={`text-sm font-bold tracking-wide ${
            isFinal ? "text-primary" : "text-foreground"
          }`}
        >
          {section.title}
        </h4>
        <ChevronDown
          size={16}
          className={`text-muted-foreground transition-transform duration-200 ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>
      {isOpen && (
        <div className="px-4 py-3 space-y-4 bg-card/50">
          {section.subsections.map((sub, i) => (
            <div key={i} className="space-y-2">
              <div className="flex items-center gap-2">
                <h5 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  {sub.title}
                </h5>
                {sub.severity && <SeverityBadge severity={sub.severity} />}
              </div>
              <ul className="space-y-1.5">
                {sub.content.map((line, j) => (
                  <li
                    key={j}
                    className="text-sm text-secondary-foreground leading-relaxed pl-3 border-l-2 border-border"
                  >
                    {line}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
