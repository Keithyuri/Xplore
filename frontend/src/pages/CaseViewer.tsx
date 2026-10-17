import { useState } from "react";
import { useCase } from "@/hooks/useCase";
import AppHeader from "@/components/layout/AppHeader";
import { LoadingScreen, ErrorScreen } from "@/components/layout/StateScreens";
import XRayViewer from "@/components/xray/XRayViewer";
import OverlayPanel from "@/components/xray/OverlayPanel";
import ReportPanel from "@/components/report/ReportPanel";
import { ChevronDown, ChevronUp, Layers } from "lucide-react";

interface CaseViewerProps {
  caseId?: string;
}

export default function CaseViewer({ caseId }: CaseViewerProps) {
  const {
    caseData,
    appState,
    reportMode,
    setReportMode,
    toggleOverlay,
    setOverlayOpacity,
    uploadAndSegment,
    isUploading,
    displayImageUrl,
  } = useCase(caseId);

  const [mobileOverlaysOpen, setMobileOverlaysOpen] = useState(false);

  if (appState === "loading") {
    return (
      <>
        <AppHeader />
        <LoadingScreen />
      </>
    );
  }

  if (appState === "error") {
    return (
      <>
        <AppHeader />
        <ErrorScreen onRetry={() => window.location.reload()} />
      </>
    );
  }

  const hasCase = !!caseData;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background">
      <AppHeader caseTitle={caseData?.caseTitle} />

      <div className="hidden flex-1 overflow-hidden lg:flex">
        <div className="flex-1 min-w-0 border-r border-border">
          <XRayViewer
            imageUrl={displayImageUrl}
            overlays={caseData?.overlays ?? []}
            onUploadImage={(file) => uploadAndSegment(file)}
            isUploading={isUploading}
          />
        </div>

        {hasCase && (
          <div className="flex w-[420px] flex-col overflow-hidden xl:w-[480px]">
            <div className="shrink-0 max-h-[40%] overflow-y-auto border-b border-border p-3">
              <OverlayPanel
                overlays={caseData.overlays}
                onToggle={toggleOverlay}
                onOpacityChange={setOverlayOpacity}
              />
            </div>

            <div className="min-h-0 flex-1 p-3">
              <ReportPanel
                report={caseData.report}
                mode={reportMode}
                onModeChange={setReportMode}
              />
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-1 flex-col overflow-y-auto lg:hidden">
        <div className="h-[45vh] min-h-[280px] shrink-0 border-b border-border">
          <XRayViewer
            imageUrl={displayImageUrl}
            overlays={caseData?.overlays ?? []}
            onUploadImage={(file) => uploadAndSegment(file)}
            isUploading={isUploading}
          />
        </div>

        {hasCase && (
          <>
            <button
              onClick={() => setMobileOverlaysOpen(!mobileOverlaysOpen)}
              className="flex items-center justify-between border-b border-border bg-card px-4 py-3"
            >
              <div className="flex items-center gap-2">
                <Layers size={16} className="text-primary" />
                <span className="text-sm font-semibold text-foreground">Overlays</span>
                <span className="text-xs text-muted-foreground">
                  ({caseData.overlays.filter((o) => o.visible).length} active)
                </span>
              </div>
              {mobileOverlaysOpen ? (
                <ChevronUp size={16} className="text-muted-foreground" />
              ) : (
                <ChevronDown size={16} className="text-muted-foreground" />
              )}
            </button>

            {mobileOverlaysOpen && (
              <div className="border-b border-border bg-card px-3 py-2">
                <OverlayPanel
                  overlays={caseData.overlays}
                  onToggle={toggleOverlay}
                  onOpacityChange={setOverlayOpacity}
                />
              </div>
            )}

            <div className="min-h-[300px] flex-1 p-3">
              <ReportPanel
                report={caseData.report}
                mode={reportMode}
                onModeChange={setReportMode}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}