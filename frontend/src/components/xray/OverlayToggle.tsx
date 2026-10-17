import type { Overlay } from "@/types/case";
import { Slider } from "@/components/ui/slider";

interface OverlayToggleProps {
  overlay: Overlay;
  onToggle: (id: string) => void;
  onOpacityChange: (id: string, opacity: number) => void;
}

export default function OverlayToggle({ overlay, onToggle, onOpacityChange }: OverlayToggleProps) {
  return (
    <div className="group">
      <button
        onClick={() => onToggle(overlay.id)}
        className="flex items-center justify-between w-full px-3 py-3 rounded-lg hover:bg-surface-hover transition-colors"
      >
        <div className="flex items-center gap-3">
          <div
            className="w-3 h-3 rounded-full transition-all duration-200"
            style={{
              backgroundColor: overlay.visible ? overlay.color : "hsl(var(--toggle-track))",
              boxShadow: overlay.visible ? `0 0 8px ${overlay.color}60` : "none",
            }}
          />
          <span className="text-sm font-medium text-secondary-foreground">
            {overlay.name}
          </span>
        </div>
        {/* Custom toggle track */}
        <div
          className="relative w-10 h-5 rounded-full transition-colors duration-200"
          style={{
            backgroundColor: overlay.visible
              ? "hsl(var(--toggle-active))"
              : "hsl(var(--toggle-track))",
          }}
        >
          <div
            className="absolute top-0.5 w-4 h-4 rounded-full bg-foreground transition-transform duration-200 shadow-sm"
            style={{
              transform: overlay.visible ? "translateX(22px)" : "translateX(2px)",
            }}
          />
        </div>
      </button>
      {/* Opacity slider (shown when active) */}
      {overlay.visible && (
        <div className="px-3 pb-3 pt-0">
          <div className="flex items-center gap-3 pl-6">
            <span className="text-xs text-muted-foreground w-14 shrink-0">Opacity</span>
            <Slider
              value={[overlay.opacity * 100]}
              onValueChange={([v]) => onOpacityChange(overlay.id, v / 100)}
              min={0}
              max={100}
              step={5}
              className="flex-1"
            />
            <span className="text-xs text-muted-foreground w-8 text-right">
              {Math.round(overlay.opacity * 100)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
