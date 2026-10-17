import { useState, useCallback, useEffect } from "react";
import type { CXRCase, AppState } from "@/types/case";
import { fetchCaseById, createCaseFromUpload } from "@/services/api";

export function useCase(caseId?: string) {
  const [caseData, setCaseData] = useState<CXRCase | null>(null);
  const [appState, setAppState] = useState<AppState>("empty");
  const [reportMode, setReportMode] = useState<"raw" | "technical">("technical");
  const [isUploading, setIsUploading] = useState(false);
  const [previewImageUrl, setPreviewImageUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!caseId) {
      setCaseData(null);
      setPreviewImageUrl(null);
      setAppState("empty");
      return;
    }

    setAppState("loading");

    fetchCaseById(caseId)
      .then((data) => {
        if (data) {
          setCaseData(data);
          setAppState("ready");
        } else {
          setCaseData(null);
          setAppState("empty");
        }
      })
      .catch((error) => {
        console.error("Failed to load case:", error);
        setAppState("error");
      });
  }, [caseId]);

  const toggleOverlay = useCallback((overlayId: string) => {
    setCaseData((prev) => {
      if (!prev) return prev;

      const current = prev.overlays;

      const isVisible = (id: string) =>
        current.find((o) => o.id === id)?.visible ?? false;

      const toggleGroup = (groupIds: string[]) => {
        const allGroupVisible = groupIds.every((id) => isVisible(id));

        return {
          ...prev,
          overlays: current.map((overlay) =>
            groupIds.includes(overlay.id)
              ? { ...overlay, visible: !allGroupVisible }
              : overlay
          ),
        };
      };

      if (overlayId === "rotation") {
        return toggleGroup(["clavicles", "spinous_process"]);
      }

      if (overlayId === "lung_fields") {
        return toggleGroup(["lung_fields", "scapula"]);
      }

      if (overlayId === "centering") {
        return toggleGroup(["centering", "scapula"]);
      }

      return {
        ...prev,
        overlays: current.map((overlay) =>
          overlay.id === overlayId
            ? { ...overlay, visible: !overlay.visible }
            : overlay
        ),
      };
    });
  }, []);

  const setOverlayOpacity = useCallback((overlayId: string, opacity: number) => {
    setCaseData((prev) => {
      if (!prev) return prev;

      return {
        ...prev,
        overlays: prev.overlays.map((overlay) =>
          overlay.id === overlayId ? { ...overlay, opacity } : overlay
        ),
      };
    });
  }, []);

  const setImageUrl = useCallback((url: string) => {
    setCaseData((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        imageUrl: url,
      };
    });
  }, []);

  const uploadAndSegment = useCallback(async (file: File) => {
    const localPreviewUrl = URL.createObjectURL(file);

    setPreviewImageUrl(localPreviewUrl);
    setIsUploading(true);
    setAppState("ready");

    setCaseData((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        imageUrl: localPreviewUrl,
        overlays: prev.overlays.map((overlay) => ({
          ...overlay,
          visible: false,
        })),
      };
    });

    try {
      const uploadedCase = await createCaseFromUpload(file);

      setCaseData({
        ...uploadedCase,
        imageUrl: localPreviewUrl,
      });

      setPreviewImageUrl(null);
      setReportMode("technical");
      setAppState("ready");
    } catch (error) {
      console.error("Upload/segmentation failed:", error);
      setAppState("error");
      throw error;
    } finally {
      setIsUploading(false);
    }
  }, []);

  const activeOverlays = caseData?.overlays.filter((o) => o.visible) ?? [];
  const displayImageUrl = previewImageUrl ?? caseData?.imageUrl ?? "";

  return {
    caseData,
    appState,
    reportMode,
    setReportMode,
    toggleOverlay,
    setOverlayOpacity,
    setImageUrl,
    uploadAndSegment,
    activeOverlays,
    isUploading,
    displayImageUrl,
  };
}