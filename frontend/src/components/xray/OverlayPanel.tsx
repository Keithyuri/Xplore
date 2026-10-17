import { Layers } from "lucide-react";
import type { Overlay } from "@/types/case";
import OverlayToggle from "./OverlayToggle";

interface OverlayPanelProps {
  overlays: Overlay[];
  onToggle: (id: string) => void;
  onOpacityChange: (id: string, opacity: number) => void;
}

export default function OverlayPanel({ overlays, onToggle, onOpacityChange }: OverlayPanelProps) {
  return (
    <div className="bg-card rounded-xl border border-border overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border">
        <Layers size={16} className="text-primary" />
        <h3 className="text-sm font-semibold text-foreground">Segmentation Overlays</h3>
      </div>
      <div className="divide-y divide-border/50">
        {overlays.map((overlay) => (
          <OverlayToggle
            key={overlay.id}
            overlay={overlay}
            onToggle={onToggle}
            onOpacityChange={onOpacityChange}
          />
        ))}
      </div>
    </div>
  );
}
