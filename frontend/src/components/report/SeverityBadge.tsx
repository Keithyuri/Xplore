interface SeverityBadgeProps {
  severity: "mild" | "moderate" | "severe";
}

const severityConfig = {
  mild: {
    label: "Mild",
    className: "bg-severity-mild/15 text-severity-mild border-severity-mild/30",
  },
  moderate: {
    label: "Moderate",
    className: "bg-severity-moderate/15 text-severity-moderate border-severity-moderate/30",
  },
  severe: {
    label: "Severe",
    className: "bg-severity-severe/15 text-severity-severe border-severity-severe/30",
  },
};

export default function SeverityBadge({ severity }: SeverityBadgeProps) {
  const config = severityConfig[severity];
  return (
    <span
      className={`inline-flex items-center text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border ${config.className}`}
    >
      {config.label}
    </span>
  );
}
