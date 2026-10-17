import { Stethoscope, BookOpen } from "lucide-react";

interface AppHeaderProps {
  caseTitle?: string;
}

export default function AppHeader({ caseTitle }: AppHeaderProps) {
  return (
    <header className="flex items-center justify-between px-4 lg:px-6 py-3 border-b border-border bg-card/80 backdrop-blur-sm">
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/20">
          <Stethoscope size={18} className="text-primary" />
        </div>
        <div>
          <h1 className="text-sm font-bold text-foreground tracking-tight">Xplore</h1>
          <p className="text-[10px] text-muted-foreground uppercase tracking-widest">
            Chest X-Ray Learning Tool
          </p>
        </div>
      </div>
      {caseTitle && (
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface border border-border">
          <BookOpen size={14} className="text-primary" />
          <span className="text-xs font-medium text-secondary-foreground">{caseTitle}</span>
        </div>
      )}
      <div className="flex items-center gap-1 text-[10px] text-muted-foreground uppercase tracking-widest">
        <span className="px-2 py-1 rounded bg-accent/20 text-accent">Educational Use Only</span>
      </div>
    </header>
  );
}
