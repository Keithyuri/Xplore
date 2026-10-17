import { Loader2, AlertTriangle, Inbox } from "lucide-react";

export function LoadingScreen() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
      <Loader2 size={32} className="text-primary animate-spin" />
      <p className="text-sm text-muted-foreground">Loading case...</p>
    </div>
  );
}

export function ErrorScreen({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
      <div className="w-12 h-12 rounded-full bg-destructive/15 flex items-center justify-center">
        <AlertTriangle size={24} className="text-destructive" />
      </div>
      <p className="text-sm text-muted-foreground">Failed to load case data.</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-4 py-2 text-xs font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyScreen() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
      <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
        <Inbox size={24} className="text-muted-foreground" />
      </div>
      <p className="text-sm text-muted-foreground">No case found.</p>
    </div>
  );
}
